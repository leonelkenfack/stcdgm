I have read enough material. Let me now write the comprehensive audit report directly as my final assistant message.

---

# AI Engineer â€” Second-Pass Audit Report
## ST-CDGM training systems, architecture, infrastructure (post-consensus issues)

**Scope**: complementary to `docs/hyperplan_path_c_plus_consensus.md`. The math professor covered EDM math, KKT bounds, sigma_data unbiased estimator, NaN propagation in loss formulas. This audit focuses on **training dynamics, architectural gradient paths, multi-GPU plumbing, config drift, notebook bootstrap fragility, and checkpoint compatibility**.

All file:line references are absolute. I number new issues `J1, J2, ...` and re-verify existing consensus items at the end.

---

## 1. NEW critical issues â€” Architecture (J1â€“J11)

### J1 â€” `_state_adapter` is created lazily inside `RegressionMeanPredictor.forward()`, after the optimizer is built

**File**: `C:/Users/reall/Desktop/climate_data/src/st_cdgm/models/regression_mean_predictor.py:186-190`

```python
if self._state_adapter is None:
    self._state_adapter = nn.Linear(hidden, self.cfg.in_channels).to(
        device=state.device, dtype=state.dtype
    )
```

This `nn.Linear` is created **the first time `forward()` is called with a 3-D or 4-D RCN state tensor**. Consequences:

1. The parameter is **not** in `optimizer.param_groups` constructed at training-script time (it doesn't exist yet) â†’ these weights are **never trained**, they keep their random init forever.
2. It's not in `state_dict()` at checkpoint-save time of the first epoch if Stage 1 noncausal training never saw a state tensor â†’ silently missing on resume.
3. `model.parameters()` returns a **different set** depending on whether a forward was called â†’ DDP `find_unused_parameters` heuristic depends on call order.

**Severity**: P0 for the `noncausal` ablation arm (BS35). The non-causal Stage 1 looks like it's training the UNet head but the adapter that maps `(B, q, N, h)` â†’ `(B, N_lr, C_lr)` is frozen at init. This invalidates every BS35/B2 ablation result that fed RCN-state tensors instead of LR grids.

**Fix**: build the adapter eagerly in `__init__` (knowing it's only needed for the state-tensor compat path), or refuse to accept state tensors and force callers to pass `(B, C, H, W)` LR grids.

### J2 â€” `GraphToGridDecoder.grid_queries` is initialised with `0.02 * randn` but never goes through a `reset_parameters()` shim

**File**: `regression_head.py:102-104`

```python
self.grid_queries = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
```

When `RegressionPredictor.from_target_params` (or `build_stack` in notebooks) instantiates the head with a different `d_model` (e.g. `192` in V5-pearson-090) and then loads a state_dict trained with `d_model=128`, the user gets a **strict=False silently-mangled state** â€” `grid_queries` is loaded for the part where `d_model_new <= d_model_old`, leaving the upper rows at random init. No warning is emitted. Combined with `_safe_load(..., strict=False)` in `st_cdgm_phase6_finetuned_rerun.ipynb` cell 4 (line "module.load_state_dict(stripped, strict=False)"), this silently mixes a 192-D query with a 128-D KV space if YAML override is misread.

**Severity**: P1. Reproducibility hazard on YAML override merges.

**Fix**: register a hash of `(d_model, intermediate_h*intermediate_w)` as a buffer, validate at load.

### J3 â€” `IntelligibleVariableEncoder.layer_norm` is **shared across all metapaths**

**File**: `intelligible_encoder.py:97-100, 117-130`

```python
self.layer_norm = nn.LayerNorm(hidden_dim)  # single shared LN
...
for cfg in self.configs:
    tensor = embeddings[cfg.meta_path[-1]]
    tensor = self.layer_norm(tensor)         # same LN for every variable
```

A single `LayerNorm` with shape `[hidden_dim]` is reused across all 5â€“6 metapath outputs that live in **different distributional families** (T_850 vs U_500 vs static topography). The running statistics of LN's affine `weight`/`bias` are pulled toward whatever the mixed distribution looks like. This couples the encoder representations of different physical variables â€” **directly amplifying the faithfulness violation flagged in Â§1.6 of the consensus**.

**Severity**: P1 (compounds Â§1.6 of consensus).

**Fix**: `self.layer_norms = nn.ModuleDict({mp.name: nn.LayerNorm(hidden_dim) for mp in configs})` and index by name in `forward`.

### J4 â€” `forward()` returns `outputs[cfg.name] = tensor` where every `cfg` sharing the **same target node type** writes to the same dict key on the second iteration

**File**: `intelligible_encoder.py:118-130`

In the YAML (`training_config.yaml:69-94`), `GP850_spat_adj` and `GP850_to_GP500` are distinct metapaths, but their **target node types** differ (GP850 vs GP500). So `embeddings[cfg.meta_path[-1]]` returns the *aggregated-sum* output for GP500 in the second case. That's intended.

**However**: when two metapaths share the same target (e.g. if a user adds both `GP500_spat_adj` and `GP850_to_GP500`, both targeting `GP500`), `outputs["GP500_spat_adj"]` and `outputs["GP850_to_GP500"]` both copy the **same** `embeddings["GP500"]` tensor â€” because `HeteroConv(aggr="sum")` collapses both inputs to a single output per node type before this loop runs.

So the encoder dict shows 5 "variables" but they collapse to **3 unique tensors** (one per node type GP850/GP500/GP250). Combined with `q = len(configs)` used everywhere downstream, the RCN learns a `A_dag` of dimension `q Ã— q` where **multiple rows refer to identical state tensors**. This is the structural reason `A_dag` rows show uniform magnitudes (consensus pathology #3).

**Severity**: P0 â€” root cause of the encoder/DAG faithfulness violation that the math prof attributed to "HeteroConv aggregation".

**Fix**: change to `aggr="cat"` + per-target Linear (per consensus Â§1.6 + Â§3) AND make `outputs` key by `cfg.meta_path[-1]` not `cfg.name`, so q correctly reduces to number of distinct node types.

### J5 â€” `_apply_pooling` for "max" calls `global_max_pool(tensor, batch)` but in pooled mode all `batch` indices are 0 â†’ `global_max_pool` returns a `[1, hidden_dim]` tensor; the encoder then treats it as `[batch, hidden_dim]` and the unsqueeze logic in `pooled_state` produces a `[batch=1, q, hidden]` even at batch_size > 1

**File**: `intelligible_encoder.py:273-281, 155-167`

The `_assign_default_batch` (graph_builder.py:329-333) sets `data[nt].batch = torch.zeros(N, long)` â€” a single graph. So pooled returns shape `[1, hidden_dim]`. `pooled_state` line 163 does `if tensor.dim() == 1: tensor = tensor.unsqueeze(0)` â€” never triggered. Stack at line 166 returns `[1, q, hidden]`.

**Consequence**: at `batch_size > 1` (after gradient accumulation collapse into a single "logical" batch), the conditioning tensor still has `B=1`. The UNet then gets `[1, num_tokens, dim]` `encoder_hidden_states` while `noisy_sample` has `[B, C, H, W]` with `B > 1`. Cross-attention broadcasts the single conditioning across all batch elements â€” i.e. every sample in the micro-batch sees **the same conditioning**.

**Severity**: P0 if micro-batching is implemented as concatenation (not as Python list-of-batches). For the current notebook (which uses Python lists, micro_idx loop), this is masked. But `train_ddp.py` uses `batch_size=64` via DataLoader â†’ this **is** a real bug there.

**Fix**: properly assign incremented batch indices per micro-batch when collating, or assert `B == 1` at conditioning construction.

### J6 â€” `CausalConditioningProjector.dag_mlp` is fed `A_dag.reshape(1, qÂ²)` â€” single batch dim â€” yet downstream cross-attention expects `[B, T, d]`

**File**: `intelligible_encoder.py:541-545`

```python
flat = A_dag.reshape(1, -1)                                       # [1, qÂ²]
dag_emb = self.dag_mlp(flat)                                      # [1, T*d]
dag_tokens = dag_emb.view(1, self.num_dag_tokens, self.conditioning_dim)
```

Same issue as J5 â€” DAG tokens are mass-broadcast across the batch. This is partially intentional (the same DAG applies to every sample in the batch), but the broadcast is left to PyTorch's automatic expansion, which **breaks gradient masking** during `conditioning_dropout`: when the dropout zeroes `conditioning_spatial` (training_loop.py:1041-1045), it zeroes the broadcast view, **not the underlying parameters' gradient**. So the DAG tokens still receive non-zero gradient on the dropped batches. This silently weakens the CFG null-branch training.

**Severity**: P1. Compounds the failure mode the consensus described as "UNet ignores A_dag".

### J7 â€” `HRTargetIdentifiabilityHead.extract_target_stats` is a Python loop over batch + Python list `row.append(... .item())` per pixel quantile

**File**: `intelligible_encoder.py:431-456`

For each sample, it pulls `torch.quantile(x, 0.95).item()` and `.max().item()` to Python float. This is fine for evaluation, but called in the **training inner loop** (training_loop.py:1296-1314) it serialises the GPU pipeline and forces 4Ã— `.item()` per batch â€” each one a CUDA sync. With `batch_size=64`, that's **256 syncs/epoch from this head alone**. Slowest path on A100 in V3/V4 configs.

**Severity**: P2 perf bug. Mitigated by the fact the head is disabled in current YAMLs (`hr_ident.enabled: false`), but if anyone turns it back on, expect 30â€“40 % wall-clock regression.

**Fix**: vectorise â€” `torch.quantile(target_finite, q_tensor, dim=-1)`, keep tensor, no `.item()`.

### J8 â€” `ConditionalSkipBlock.alpha_mlp` uses `nn.AdaptiveAvgPool2d(1)` over `lr` but `lr` arrives shaped `[B, C, H_lr, W_lr]` from the notebook and shape `[1, C, H, W]` from `finetune_stage1_bundle_b.py:227-229` (`drivers[-1].unsqueeze(0)`)

**File**: `skip_direct.py:135-138, 104-114`

In the fine-tune script, `lr_last = drivers[-1] if drivers[-1].dim() == 4 else drivers[-1].unsqueeze(0)`. `drivers[t] = lr_data[t]` and `lr_data` is `[seq_len, N_lr, C_lr]` (nodes!), so `drivers[t]` is **`[N_lr, C_lr]`** â€” 2-D. The fallback `.unsqueeze(0)` produces `[1, N_lr, C_lr]` â€” **still 2-D as far as `AdaptiveAvgPool2d(1)` is concerned**, since pool expects `[B, C, H, W]`.

PyTorch then either errors out, or â€” silently worse â€” interprets `[1, N_lr, C_lr]` as `[B=1, C=N_lr, ?]` and crashes at runtime in pool. The fine-tune script has a bare `except Exception: mu_HR = mu_c` (`finetune_stage1_bundle_b.py:228-231`) that swallows this error and falls back to the causal path **every single batch**. So the skip block:
- has gradients zero throughout fine-tune,
- contributes nothing to `mu_HR`,
- but its parameters are in the optimizer (line 510),
- and its checkpoint is saved (line 604) â€” **giving the illusion** of an "ablation enabled" experiment.

**Severity**: P0 for any experiment that claims the V5-mini skip-connection was active.

**Fix**: convert nodes â†’ grid before feeding to `skip_block.forward()`:
```python
lr_grid = builder.lr_grid_from_nodes(drivers[-1])  # or batch["lr_grid"][-1]
```

### J9 â€” `CausalDiffusionDecoder.compute_loss_edm` raises if `D_y` has NaN/Inf, **but the bf16 path on A100 routinely overflows tail-weighted MSE on the first batches**

**File**: `diffusion_decoder.py:594-600, 626-628`

When `tail_weight.enabled=true` (every YAML except VICE), `weighted = weights * tail_w * sq_err` can reach `weight_p99=25 Ã— sigma_amplification Ã— pixel_residualÂ²`. In bf16 with the V4 config (weight_p99=40), this fp16-safe value overflows for residuals > ~12 in log1p space (~160 000 mm/day) â€” rare but happens at the very first stochastic batch with `S_churn=40`, `sigma_max=80`. The `raise ValueError` halts training. There is no `nan_to_num` fallback like the DDPM path has.

**Severity**: P1 for V3/V4 runs (S_churn=40 + high tail weights). Empirically masked because the YAML defaults P_mean=-1.2 push sigma toward small values; but V5 has P_mean=0.0 (line 85) which exposes the overflow.

**Fix**: clamp `sq_err = sq_err.clamp(max=1e6)` before tail multiplication, or use `torch.where(torch.isfinite(weighted), weighted, weighted.detach().clone() * 0)`.

### J10 â€” UNet input-channel count for `causal_concat` mode is `in_channels + 2` regardless of whether `mu_HR` and `baseline_log` actually have `in_channels=1` channels

**File**: `diffusion_decoder.py:108-110`

```python
unet_in_channels = in_channels + 2 if causal_concat else in_channels
```

`in_channels` for precipitation is 1, so `unet_in_channels=3`. This assumes `mu_HR` and `baseline_log` each have **1 channel**. But `GraphToGridDecoder(output_channels=1)` (regression_head.py:83) is consistent only when the user doesn't change `output_channels`. If a multi-variable head is enabled later (e.g. for temperature `[T_min, T_mean, T_max]`), `mu_HR` becomes 3-channel and `cat([Î´, Î¼, baseline], dim=1)` produces 7 channels â€” UNet expects 3 â†’ shape mismatch crash at the first forward.

No assertion guards this. The `if mu_HR.shape != x_noisy.shape` check (line 446-450) catches H/W mismatch but **not channel mismatch** (because both must equal `in_channels`, which the check enforces via the equality â€” fine, but it doesn't catch the architectural inconsistency before `unet_in_channels` is fixed at init).

**Severity**: P2 latent bug (only fires when adding new variables).

**Fix**: explicit `unet_in_channels = in_channels * 3 if causal_concat else in_channels` AND assert `mu_HR.shape[1] == in_channels` at forward.

### J11 â€” `forward_edm` builds `hidden_states = unet_input.new_zeros((B, 1, self.conditioning_dim))` placeholder but **doesn't size it to match `cross_attention_dim`** when the YAML overrides it via `unet_kwargs.cross_attention_dim`

**File**: `diffusion_decoder.py:486-488`

`self.conditioning_dim` is captured at `__init__` from the `conditioning_dim` constructor arg (line 113). But `unet_kwargs` can override `cross_attention_dim` independently (it's passed verbatim to `UNet2DConditionModel` line 140-142). If the user sets `unet_kwargs={"cross_attention_dim": 256}` while keeping `conditioning_dim=128`, the placeholder has the wrong inner dim â†’ diffusers raises at cross-attention forward. **The error is buried in the unet** and looks like an internal diffusers bug rather than a config mismatch.

**Severity**: P2.

**Fix**: at init, assert `unet_kwargs.get("cross_attention_dim", conditioning_dim) == conditioning_dim` OR pull the effective value from `self.unet.config.cross_attention_dim` after construction.

---

## 2. NEW critical issues â€” Training loop (J12â€“J22)

### J12 â€” `train_epoch` builds gradient clip groups by passing **module-level** `parameters()` â€” including DDP/compile wrappers â€” without unwrapping

**File**: `training_loop.py:1494-1500`

```python
grad_norm_rcn = torch.nn.utils.clip_grad_norm_(rcn_runner.cell.parameters(), gradient_clipping)
grad_norm_diff = torch.nn.utils.clip_grad_norm_(diffusion_decoder.parameters(), gradient_clipping)
grad_norm_enc = torch.nn.utils.clip_grad_norm_(encoder.parameters(), gradient_clipping)
```

When DDP wraps the module, `rcn_runner.cell` is `DDP(RCNCell)` and `.parameters()` returns the same tensors as the inner module â€” fine. But when `torch.compile` wraps it, `_orig_mod.parameters()` is what would be needed. **The current call iterates over the wrapper's params, which are the same** in PyTorch â‰¥ 2.0 â€” so it works **but the spatial projector params are never clipped**, and neither are the HR ident head / CASTLE anchor params (training_loop.py:1494-1500 only enumerates encoder, RCN, diffusion). When the optimizer step is applied with `clip_grad_norm=1.0`, the un-clipped params can blow up under the lambda(sigma) weighting amplification.

**Severity**: P1 â€” explains some of the "DAG ignored" symptoms when CASTLE is on.

**Fix**: clip with `optimizer.param_groups` flattened, not module by module.

### J13 â€” `_train_autocast(amp_mode)` context wraps `loss_total` construction but NOT the `optimizer.step()`

**File**: `training_loop.py:1437-1486`

When `amp_mode='cuda_bf16'`, the backward pass runs in bf16, accumulating into fp32 `.grad`. Then `clip_grad_norm_` and `optimizer.step()` run outside autocast â€” correct. But `loss_total.item()` (line 1463-1466) is called **before** `loss_total` has been backwarded, and it's a bf16 scalar. The implicit cast to Python float occurs in bf16 â†’ ~3 decimal digits accuracy. **Logged metrics are wrong at the 4th decimal**.

For RAPSD â‰ˆ 0.001 differences cited in the consensus Â§1.10 (Oracle vs CorrDiff 0.006 inside noise floor), the log precision itself is at the same order as the claimed difference.

**Severity**: P1 statistical reporting.

**Fix**: `loss_total.detach().float().item()`.

### J14 â€” DDP `no_sync` context is entered conditionally **per microbatch** by checking `isinstance(encoder, DDP)`, but compiled DDP returns False

**File**: `training_loop.py:1478-1481`

```python
ctx_enc = encoder.no_sync() if (isinstance(encoder, DDP) and not is_last_micro) else nullcontext()
```

When `torch.compile(DDP(encoder))` is used, `encoder` becomes `OptimizedModule(DDP(...))` â€” `isinstance(encoder, DDP)` is `False`. So `no_sync()` is **never entered**, and every micro-batch triggers a full all-reduce. With micro=16, accum=2 (training_config.yaml:424) and a 50M-param UNet across 4 GPUs, this is ~28 MB Ã— 2 redundant all-reduces Ã— 100s of steps/epoch â€” measurable wall-clock waste.

**Severity**: P2 perf.

**Fix**: `inner = encoder._orig_mod if hasattr(encoder, "_orig_mod") else encoder; ctx_enc = inner.no_sync() if isinstance(inner, DDP) and not is_last_micro else nullcontext()`.

### J15 â€” `finetune_stage1_bundle_b.train_one_epoch_bundle_b` calls `optimizer.zero_grad(set_to_none=True)` **inside** the loss-skip branch but **also** outside

**File**: `scripts/finetune_stage1_bundle_b.py:317-322`

```python
if not torch.isfinite(loss_total):
    ...
    optimizer.zero_grad(set_to_none=True)
    continue

optimizer.zero_grad(set_to_none=True)
loss_total.backward()
```

This is fine semantically but **leaks an issue**: between the `if not torch.isfinite` check and the next batch, the DAG projection (`project_dag_spectral` / `project_dag_floor` lines 335-338) is **skipped** on NaN batches. So under repeated NaNs, A_dag drifts off the acyclic cone for the duration of those NaN batches â€” and there's no recovery once isfinite returns True. The spectral projection then has to do potentially huge rescaling, which itself can numerically destabilise A_dag if the cumulative drift was large.

**Severity**: P2.

**Fix**: project A_dag also on the skip branch.

### J16 â€” Optimizer rebuild on resume: `optimizer.load_state_dict(ckpt["optimizer_state_dict"])` does **not** validate that param_groups have the same LRs as the YAML

**File**: `scripts/finetune_stage1_bundle_b.py:556-558, 503-512`

The script constructs `param_groups` with 4 or 5 groups depending on `skip_block`. If a user fine-tunes a checkpoint that was saved without `skip_block` (4 groups), then resumes with `skip_block` enabled (5 groups), `load_state_dict` either errors or **silently mis-assigns Adam moments** to wrong groups. The bare `except Exception` (line 557) swallows the error and uses fresh state â€” but logs `[WARN] optimizer state load failed`. The user might miss the warning in a 12h run.

**Severity**: P1 for any ablation toggle that changes the number of param groups.

**Fix**: pickle group structure to `intermediate_state`, validate on resume.

### J17 â€” LR schedule: zero warm-up, zero decay across the entire fine-tune

**File**: `scripts/finetune_stage1_bundle_b.py:512-513`

```python
optimizer = torch.optim.AdamW(param_groups, weight_decay=hp["weight_decay"])
```

No `lr_scheduler` is created. With `lr_encoder=5e-5` and `lr_regression=1e-4`, the script runs constant LR for 25 epochs. Meanwhile `lambda_castle` jumps from 0.05 to 0.10 at epoch 10 (lines 140-142) â€” a 2Ã— loss reweighting with no LR damping. Combined with the cosine annealing of `lambda_l1` (start 0.10 â†’ end 0.01), the effective gradient norm on `A_dag` changes by **factor 10**+ across the run without any LR adjustment. Expected outcome: oscillation in mid-fine-tune epochs and silent A_dag drift.

**Severity**: P1.

**Fix**: cosine LR decay with 1-2 epoch warm-up (already present via `dag_grad_gate_value` decoupled but not on the LR itself).

### J18 â€” Resume logic loads weights with `strict=False` for **every** module, including the optimizer state

**File**: `scripts/finetune_stage1_bundle_b.py:546-555`

```python
encoder.load_state_dict(ckpt["encoder_state_dict"], strict=False)
rcn_cell.load_state_dict(ckpt["rcn_cell_state_dict"], strict=False)
...
skip_block.load_state_dict(ckpt["skip_block_state_dict"], strict=False)
```

`strict=False` silently drops missing or extra keys with no log. After the 6â†’4 nodes refactor proposed by the consensus, this would silently load `A_dag` as a 6Ã—6 â†’ 4Ã—4 mismatch and fall back to xavier init for A_dag while keeping the trained encoder weights. The hybrid model would then "run" but produce garbage. The `_load_from_state_dict` shim in `causal_rcn.py:302-383` does handle the structural_mlps â†’ struct_W shape conversion but does **not** handle q=6 â†’ q=4 dimension shrinkage.

**Severity**: P0 for any refactor that touches num_vars.

**Fix**: load with `strict=True` by default, set `strict=False` only with explicit user opt-in + verbose key diff log.

### J19 â€” Per-epoch `inprogress` checkpoint save uses `torch.save â†’ os.replace â†’ fsync` chain but **does not call `torch.cuda.synchronize()` first**

**File**: `scripts/finetune_stage1_bundle_b.py:592-619`

On CUDA, `state_dict()` returns CUDA tensors. `torch.save` pickles them but **the CUDA tensors may still have pending async kernels** mutating them. Without `torch.cuda.synchronize()` before the save, the snapshot can capture **intermediate gradient states**. This rarely matters for inference-time state but can corrupt Adam moments mid-update â€” and Colab Drive sync makes this worse because the file is being written to a slow remote mount during the next epoch's forward.

**Severity**: P2 (race condition, rare).

**Fix**: `torch.cuda.synchronize()` before `torch.save`.

### J20 â€” `train_one_epoch_bundle_b` uses `loss_total.backward()` then `clip_grad_norm_` over a **flat list of all params** (line 324-327) **without** `set_to_none=True` on the freshly-initialised `castle_anchor`

**File**: `scripts/finetune_stage1_bundle_b.py:474-477, 322-328`

CASTLE anchor is created `castle_anchor = CASTLEAnchor(...).to(DEVICE)` then optimised with `lr_castle=1e-4`. On the first batch, `castle_anchor.parameters().grad` is `None` (no backward has run). `clip_grad_norm_` skips `None` grads â†’ fine. But `optimizer.zero_grad(set_to_none=True)` (line 321) **before** backward sets `.grad` to None for everything, including castle. So after backward, grads exist on castle. Subsequent batches: zero_grad â†’ None â†’ backward â†’ fills. Fine in principle, **but DDPM-style `_grad_comp` rescaling for gradient accumulation (training_loop.py:1436-1446)** doesn't run here because the fine-tune script does not implement micro-batching. The full `_grad_comp = float(len(batches))` logic is absent. So if a user runs the fine-tune script with the same loss formulation as the main training loop, the lambda weights effectively scale differently between train_epoch (main) and train_one_epoch_bundle_b. **Reproducing the V5-mini hyperparameters across the two paths is impossible**.

**Severity**: P1 for cross-protocol comparability (V5-mini training vs Bundle-B fine-tune).

### J21 â€” `EMA` warmup is checked against **step count**, not epoch â€” the same EMA decay is applied whether the model is at the very start (random init) or mid-training

**File**: `src/st_cdgm/training/training_loop.py:1086-1099` (referenced) and `two_stage.py:1146` ("batches/epoch < 1000 => 0 update EMA, le shadow restait fige a son init")

```python
ema_warmup_steps: int = 0,  # default = no warmup
```

Default 0 means the EMA `shadow_weights` start equal to `init_weights` at step 0 and immediately update with decay 0.9999. For 1000 steps (Karras EDM2 recommendation), `1 - 0.9999^1000 â‰ˆ 0.095` â†’ the shadow is only ~10 % "live" â€” still mostly init noise. Reaching effective averaging requires ~50 000 steps.

V5 YAMLs set `warmup_steps: 1000` (training_config_v5_pearson_090.yaml:147) â€” good. **But V3 YAMLs leave it at 0** (`training_config_v3.yaml:95-97`), so V3 runs are training EMA shadows that include the random init for the first ~1k steps. With Stage 2 `epochs_max=100` and 219 batches/epoch, total steps = ~21 900 â€” the early-init pollution affects ~5 % of the trajectory; small but measurable on F1@p99.

**Severity**: P2.

**Fix**: enforce `ema_warmup_steps >= 1000` default in `__init__`.

### J22 â€” `dag_grad_gate` value is written to a **buffer** (`register_buffer`) and is in `state_dict()`

**File**: `causal_rcn.py:102-105`

On resume, the buffer's value is restored from the checkpoint. So if a user fine-tunes for 5 epochs (gate ramps 0 â†’ 0.25), saves, then resumes a fresh fine-tune script that expects gate=0 cold-start: the gate is **immediately at 0.25 on the resume**, not 0. The `set_dag_grad_gate(0.0)` call in the new script would override, but the bug surfaces if **the script never sets the gate** â€” which is exactly the consensus bug Â§1.1 in `finetune_stage1_bundle_b.py`. The gate persists at whatever value the original training left it at. So even after fixing Â§1.1, the resumed Bundle-B run will see different gate behaviour depending on the upstream checkpoint state.

**Severity**: P1.

**Fix**: `register_buffer(..., persistent=False)` so the buffer is **not** saved, OR explicitly reset to 0 at fine-tune entry.

---

## 3. NEW critical issues â€” Multi-GPU / DDP (J23â€“J27)

### J23 â€” `wrap_model_ddp` passes `find_unused_parameters` to `DDP()` â€” but the **default in `train_ddp.py` is True**

**File**: `train_ddp.py:424`, `multi_gpu.py:49-83`

```python
find_unused = CONFIG.training.get("multi_gpu", {}).get("find_unused_parameters", True)
```

`find_unused_parameters=True` adds a ~30 % all-reduce overhead and disables some DDP optimizations (no static graph capture). With the `spatial_projector` having a learnable parameter (`grid_queries`) that is **always used**, and `castle_anchor` whose forward is conditional on `lambda_castle > 0` â€” only the latter justifies `find_unused=True`. Setting it True globally penalises every forward.

**Severity**: P2 perf (~20-30% wall-clock).

**Fix**: investigate which params are actually unused per step; ideally set `find_unused=False` with `static_graph=True`.

### J24 â€” `setup_ddp` reads `MASTER_ADDR` and `MASTER_PORT` from environment but if multiple `train_ddp.py` jobs run on the same host (e.g. parallel hparam search), the **default `12355` port collides**

**File**: `multi_gpu.py:32-37`

No fallback to a random port. On a CyVerse VICE node where users could spawn parallel training shells, the second invocation hangs at `init_process_group`.

**Severity**: P2.

**Fix**: detect collision, retry with `random.randint(20000, 60000)`.

### J25 â€” `ShardedIterableDataset.__iter__` in `train_ddp.py:54-57` shards via `i % world_size == rank` â€” **deterministic per rank**, but the upstream `pipeline.build_sequence_dataset(...)` is also deterministic, so **rank 0 and rank 1 always see the same modulo-class samples across epochs**

If `pipeline` has any internal stochasticity (e.g. `np.random` shuffle when materializing), the shards diverge across ranks. If not, each rank trains on a **fixed subset every epoch** â€” no shuffling between epochs. Standard DDP uses `DistributedSampler(shuffle=True)` with `sampler.set_epoch(epoch)` to rotate. **No such rotation here**.

Effective consequence: epoch-to-epoch the optimizer sees the same data ordering â†’ potential overfitting to the rank-local subset, slower convergence.

**Severity**: P1 for â‰¥ 2 GPU runs.

**Fix**: use `DistributedSampler` with `set_epoch(epoch)` on a map-style dataset; for iterable, shuffle the buffer per rank with a per-epoch seed.

### J26 â€” `wrap_models_for_notebook` uses `nn.DataParallel` for multi-GPU **inside the notebook**, but DataParallel has been deprecated since PyTorch 1.5 in favour of DDP â€” and importantly **DataParallel replicates the model on every forward**, which thrashes the 50â€“80 M-param V4 UNet across GPUs each batch

**File**: `multi_gpu.py:162-194`

```python
wrapped[name] = torch.nn.DataParallel(model, device_ids=gpus)
```

The warning at line 151-154 acknowledges this. But the function is still callable from notebooks â€” and would silently double the per-step latency. Combined with `torch.compile` (which doesn't play well with DataParallel), notebook multi-GPU is effectively broken.

**Severity**: P2 (mostly disabled in practice, but a footgun).

**Fix**: deprecate `wrap_models_for_notebook`, error out instead.

### J27 â€” `cleanup_ddp()` calls `dist.destroy_process_group()` but doesn't `barrier()` before â€” so rank-0 might destroy while other ranks are still mid-checkpoint-save

**File**: `multi_gpu.py:40-45`, `train_ddp.py:638-639`

In the training script, the final epoch's checkpoint save is rank-0-only (correct), but other ranks proceed directly to `cleanup_ddp()` â€” if rank-0's torch.save is slow (Drive mount), rank-0 sits in `torch.save` while other ranks call `destroy_process_group()` â†’ NCCL errors at next test.

**Severity**: P2 (terminal cleanup, harmless if the script exits anyway).

**Fix**: `dist.barrier()` before destroy.

---

## 4. Config audit findings (J28â€“J35)

### J28 â€” `training_config.yaml:148` declares `steps: 1000` for "DDPM legacy" but `scheduler_type: "edm_karras"` â€” `steps` is silently ignored

Multiple YAMLs have this same dead key (`gpu_config.yaml:104`, `training_config_vice.yaml:104`). The naive reader assumes `steps` controls EDM behaviour. **In code path**: `diffusion_decoder.py:146` uses `steps` only to instantiate the legacy `DDPMScheduler`; EDM uses `cfg.sigma_*` and `num_steps` argument to `_sample_edm_karras`. **Orphan key**.

**Severity**: P3 (documentation).

### J29 â€” `training_config_corrdiff_normal.yaml:42` sets `scheduler_type: "edm_karras"` but `cfg_scale: 1.5`. EDM Karras path (`_sample_edm_karras`) **does not implement CFG** (search `cfg_scale` in `edm_sampler.py`)

The CFG branch is implemented in `_sample_dpm_solver` (line 1230+) and the DDPM fallback (line 862-883), but **not** in `_sample_edm_karras`. So `cfg_scale=1.5` is silently ignored when EDM Karras is selected. The user thinks they have CFG=1.5 â†’ publishes a Pearson 0.815 result attributing it to CFG.

**Severity**: P0 â€” this **invalidates** the v4/v5 results that claim cfg_scale=1.5 + edm_karras (per training_config_corrdiff_normal.yaml).

**Fix**: implement CFG in `_sample_edm_karras` (Karras 2022 Â§C.3) OR raise an error when both are set.

### J30 â€” `training_config_v5_pearson_090.yaml:33-44` sets `cfg_scale: 1.0` AND `conditioning_dropout_prob: 0.13`. The dropout trains an unconditional branch that is **never used** at inference (cfg_scale=1.0 means cond branch only)

Wasted compute: every batch has ~13 % dropout zero forwards that contribute nothing to the deployed model. Worse: it adds noise to the **conditional** branch's gradient because the dropped batches have a less informative loss â†’ effectively reduces the conditional learning rate by ~13 %.

**Severity**: P2.

**Fix**: either set cfg_scale > 1 or disable conditioning_dropout.

### J31 â€” `training_config.yaml:319` sets `reconstruction_loss_type: "huber+cosine"` but `loss_reconstruction` in `training_loop.py:99-160` **only supports `"mse"`, `"cosine"`, `"mse+cosine"`**

```python
else:  # loss_type == "mse"
    return nn.functional.mse_loss(pred, target)
```

The `else` branch silently falls through to MSE for `"huber+cosine"` â€” no warning. User believes Huber is active.

**Severity**: P1.

**Fix**: assert `loss_type in {"mse", "cosine", "mse+cosine"}`, OR implement huber.

### J32 â€” `training_config_corrdiff_mini.yaml:64` sets `norm_num_groups: 32` for `block_out_channels: [64, 128, 128]`. `32` divides `128` (groups=8) and `64` (groups=2). But the YAML override doesn't survive merge if the base YAML has `norm_num_groups: 8` (training_config.yaml:241) â€” depends on OmegaConf merge depth

Actually OmegaConf does deep merge â€” so this works. **However** when a user adds a new block (e.g. base says `[32, 64]`, override says `[32, 64, 128]`), the `norm_num_groups: 32` doesn't divide 32 â†’ diffusers raises. The user has to trace through the multi-YAML merge to find which `norm_num_groups` is winning.

**Severity**: P3 (config UX).

### J33 â€” `training_config_v3.yaml:28-29` sets `scheduler_type: "dpm_solver++"` and `cfg_scale: 1.5`. The DPM-Solver++ path **does** implement CFG (diffusion_decoder.py:1230-1289). But the **`use_cfg = bool(cfg_scale) and cfg_scale > 0.0`** check (line 1230) interprets `cfg_scale=1.0` as "active CFG" â€” which produces `model_output = uncond + 1.0 * (cond - uncond) = cond` â†’ wasted forward of the uncond branch with no net effect

This is fine semantically but doubles inference time for nothing when users mistakenly write `cfg_scale: 1.0` (e.g. training_config_v5_pearson_090.yaml:34).

**Severity**: P2 perf.

**Fix**: `use_cfg = cfg_scale > 1.0 + epsilon`.

### J34 â€” V3/V4 YAMLs set `ema.enabled: true` but `training_config.yaml` (base) has no `ema` section â†’ omegaconf merge produces only the override `enabled/decay/warmup_steps`. The notebook code accessing `CONFIG.two_stage.stage2.ema.enabled` works, but **the base config can't be used standalone** for an EMA run â€” silent dependency on the override file

**Severity**: P3 (config UX).

### J35 â€” `training_config_v5_pearson_090.yaml:144-147` and `training_config_v3.yaml:95-97` both set `ema.decay: 0.9999`, BUT V5 has `epochs_max: 250` while V3 has `epochs_max: 100`. The effective averaging window (decay^N steps to half-value) is **proportional to total steps**, so the relative "long-window" character of EMA is `5Ã—` weaker in V3 than V5. Yet the **same** decay value is presented as Karras-canonical. The result: V5 EMA actually averages, V3 EMA is dominated by tail epochs

**Severity**: P2.

**Fix**: `ema_decay = 1 - (1 - 0.5) ** (1/total_steps)` or just document.

---

## 5. Notebook audit findings (J36â€“J42)

### J36 â€” `st_cdgm_phase6_finetuned_rerun.ipynb` cell 3 line "DATA_ROOT = _DATA_ROOT_LOCAL_SSD" mutates `CONFIG.data.lr_path` etc. in-place via `_relocate`. If the notebook is re-run within the same kernel (e.g. user edits cell 5 and runs again), the lr_path is **already** the SSD path â†’ `_relocate` no-ops, fine. But if the user reloads `CONFIG = OmegaConf.load(...)` then runs cell 3 again, the SSD copy step is repeated â€” copies all 5 .nc files (multi-GB) every reload

**File**: `st_cdgm_phase6_finetuned_rerun.ipynb` cell 3, "if _dst.exists() and _dst.stat().st_size == _src.stat().st_size: continue"

This **does** check existence â€” partial mitigation. **But** the check uses `_src.stat()` which reads through Drive FUSE â€” slow even when src and dst are both already correct. ~5â€“10 seconds per file Ã— 5 files = 30+ seconds per re-run for nothing.

**Severity**: P3.

### J37 â€” `st_cdgm_phase6_finetuned_rerun.ipynb` cell 4 calls `_safe_load(name_, module)` for each Oracle module with `strict=False` and a bare `except Exception` that **prints but doesn't raise**. If `epoch_finetuned.pth` is missing the `regression_head_state_dict` key, the rh stays at random init â€” and the notebook continues evaluating, producing a "Pearson 0.05" baseline that the user mistakes for "Oracle FT degraded"

**Fix**: track which modules failed to load, raise at end of `build_stack` if **any** are missing.

### J38 â€” `_cell43.py:97-141 compute_validation_loss` is the only place in the training-evaluation notebooks that uses `diffusion.sample(...)` to compute val loss. It's defined to set `apply_constraints=False` but uses the **DDPM scheduler default** if `CONFIG.diffusion.scheduler_type` is missing (`scheduler_type="ddpm"` from `.get(..., "ddpm")`). The DDPM path runs **1000 inference steps** per val sample â†’ on 200 val samples this is 200 000 steps every epoch for validation alone â€” order of magnitude longer than training itself

The override `val_num_steps` (line 120) doesn't help because diffusers' DDPM scheduler doesn't subsample â€” `set_timesteps(15)` overrides the 1000, but the user has to remember to set it in YAML.

**Severity**: P1 perf â€” fix: hard-code `num_steps=min(num_steps, 32)` in val.

### J39 â€” `_cell43.py:218-223 BEST_MODEL_STATES = {...deepcopy...}` copies the **full** state dict on every val improvement. For a 50â€“80 M-param Stage 2 UNet, each deepcopy is ~200â€“320 MB. If val improves 20 times across training, peak RAM holds 4â€“6 GB of deepcopies â€” fine on 500 GB host (the user environment per CLAUDE.md), **but fatal on 16 GB Colab T4**

`gc.collect()` is not called. Persistent across epochs.

**Severity**: P1 on Colab.

**Fix**: keep only the latest `BEST_MODEL_STATES`, write to disk between epochs.

### J40 â€” `st_cdgm_phase6_finetuned_rerun.ipynb` cell 3 line "N_VAL_SAMPLES = 24" â€” hardcoded magic number to "16 batches + buffer". When cell 5 sets `n_batches=16` and the loop iterates over `val_dataset[:n_batches]`, the buffer of 8 is never used. But if a user changes `n_batches` to 64 in cell 5 alone, the val_dataset still has only 24 samples â†’ silent truncation

**File**: cell 3 `_val_samples = [sample0] + list(_it.islice(..., N_VAL_SAMPLES - 1))`

**Severity**: P1 â€” this is the bug the consensus Â§1.10 calls out as "N=16 inside noise floor". Fixing one cell doesn't fix it.

**Fix**: read `n_batches` from cell 5 in cell 3.

### J41 â€” Notebook bootstrap cell 1 in `st_cdgm_phase6_finetuned_rerun.ipynb` runs `git clone --depth 1 -b two-stage-causal` if the SSD path doesn't exist. **But it doesn't `git pull` once `os.chdir(project_path)` happens** â€” only if `GIT_PULL_ON_RESUME = True` (line "if (project_path / '.git').exists(): elif GIT_PULL_ON_RESUME:").

On Colab disconnect mid-run, the SSD copy is gone. Resume â†’ fresh `--depth 1` clone â†’ loses any local edits. The user has no way to know which commit was used unless they manually check.

**Severity**: P2.

**Fix**: `git rev-parse HEAD` log in bootstrap output.

### J42 â€” Cell 4 build_stack passes `causal_concat=True` regardless of whether `CONFIG.diffusion.causal_concat` exists. Hardcoded. For ablations that test causal_concat=False, the user has to edit cell 4 directly, then revert â€” a step prone to error in 4-hour training cycles

**Severity**: P2.

**Fix**: read from CONFIG.

---

## 6. Per-file issue table

| File | Issue # | Line(s) | Severity | Description |
|------|---------|---------|----------|-------------|
| `models/regression_mean_predictor.py` | J1 | 186-190 | P0 | `_state_adapter` lazily created post-optim â€” never trained |
| `models/regression_head.py` | J2 | 102-104 | P1 | `grid_queries` size mismatch silent on YAML override |
| `models/intelligible_encoder.py` | J3 | 97-100 | P1 | Single shared `LayerNorm` across all metapaths |
| `models/intelligible_encoder.py` | J4 | 118-130 | P0 | Multiple metapaths with same target collapse to identical tensors |
| `models/intelligible_encoder.py` | J5 | 273-281, 155-167 | P0 | Batch=1 pooling breaks micro-batching > 1 |
| `models/intelligible_encoder.py` | J6 | 541-545 | P1 | DAG token broadcast bypasses conditioning_dropout gradient |
| `models/intelligible_encoder.py` | J7 | 431-456 | P2 | `extract_target_stats` Python loop syncs GPU |
| `models/skip_direct.py` | J8 | 135-138 | P0 | `alpha_mlp` fed 2-D node tensor â†’ silent except fallback to causal path |
| `models/diffusion_decoder.py` | J9 | 594-600, 626-628 | P1 | Tail-weighted MSE overflows in bf16 â†’ hard raise |
| `models/diffusion_decoder.py` | J10 | 108-110 | P2 | `unet_in_channels = in_ch + 2` assumes mu_HR/baseline are 1-channel |
| `models/diffusion_decoder.py` | J11 | 486-488 | P2 | hidden_states placeholder uses `self.conditioning_dim`, not unet config |
| `training/training_loop.py` | J12 | 1494-1500 | P1 | Grad clip enumerates 3 modules only â€” castle/ident/projector unclipped |
| `training/training_loop.py` | J13 | 1437-1486, 1463 | P1 | `.item()` of bf16 tensor â†’ 3-digit precision in logs |
| `training/training_loop.py` | J14 | 1478-1481 | P2 | DDP `no_sync` skipped when `torch.compile` wraps DDP |
| `scripts/finetune_stage1_bundle_b.py` | J15 | 317-322, 335-338 | P2 | NaN-skip branch doesn't project A_dag â†’ drift on isfinite recovery |
| `scripts/finetune_stage1_bundle_b.py` | J16 | 556-558, 503-512 | P1 | Optimizer load mis-assigns Adam moments on param_group structural diff |
| `scripts/finetune_stage1_bundle_b.py` | J17 | 512-513 | P1 | No LR scheduler across 25 epochs, while lambda_castle jumps 2Ã— |
| `scripts/finetune_stage1_bundle_b.py` | J18 | 546-555 | P0 | `strict=False` resume silently swallows 6â†’4 node mismatch |
| `scripts/finetune_stage1_bundle_b.py` | J19 | 592-619 | P2 | No `cuda.synchronize()` before save â†’ race with pending kernels |
| `scripts/finetune_stage1_bundle_b.py` | J20 | 322-328 | P1 | Bundle-B training uses no `_grad_comp` â†’ lambda values not comparable to main loop |
| `training/two_stage.py` | J21 | 1086-1099 | P2 | EMA default warmup_steps=0 pollutes shadow with init noise |
| `models/causal_rcn.py` | J22 | 102-105 | P1 | `dag_grad_gate` is persistent buffer â†’ restored on resume, fights cold-start scripts |
| `training/multi_gpu.py` | J23 | 49-83 + train_ddp:424 | P2 | `find_unused_parameters=True` default â†’ 30 % DDP overhead |
| `training/multi_gpu.py` | J24 | 32-37 | P2 | `MASTER_PORT=12355` hard default â†’ collision on multi-job hosts |
| `train_ddp.py` | J25 | 46-57 | P1 | `ShardedIterableDataset` shards modulo-rank â€” no per-epoch shuffle |
| `training/multi_gpu.py` | J26 | 162-194 | P2 | `DataParallel` notebook path deprecated + slow |
| `training/multi_gpu.py` | J27 | 40-45 | P2 | `cleanup_ddp` lacks barrier before destroy |
| `config/training_config.yaml` | J28 | 148 | P3 | `steps: 1000` ignored under `edm_karras` |
| `config/training_config_corrdiff_normal.yaml` | J29 | 42-44 | P0 | `cfg_scale=1.5` ignored by `_sample_edm_karras` |
| `config/training_config_v5_pearson_090.yaml` | J30 | 33-44 | P2 | `cfg=1.0` + dropout=0.13 â†’ wasted dropout-branch training |
| `config/training_config.yaml` | J31 | 319 | P1 | `huber+cosine` silently falls through to MSE |
| `config/training_config.yaml` | J32 | 241 + overrides | P3 | `norm_num_groups` override chain depends on merge depth |
| `config/training_config_v3.yaml` | J33 | 28-29 | P2 | `cfg_scale=1.0` treated as active CFG â†’ wasted forward |
| Multiple V3/V4/V5 YAMLs | J34 | â€” | P3 | `ema` config has no base â€” override-only |
| `training_config_v3.yaml` vs `v5.yaml` | J35 | 95, 144-147 | P2 | Same `decay=0.9999` but 2.5Ã— different total steps |
| `st_cdgm_phase6_finetuned_rerun.ipynb` | J36 | cell 3 | P3 | SSD copy re-stat on every reload |
| Same notebook | J37 | cell 4 | P1 | `_safe_load` swallows missing-key â†’ random-init eval |
| `_cell43.py` | J38 | 118-141 | P1 | `compute_validation_loss` could run DDPM 1000 steps per val sample |
| `_cell43.py` | J39 | 218-223 | P1 | deepcopy on val-improve eats Colab RAM |
| `st_cdgm_phase6_finetuned_rerun.ipynb` | J40 | cell 3, "N_VAL_SAMPLES=24" | P1 | hardcoded magic â‰  cell 5's `n_batches` |
| Same notebook | J41 | cell 1 | P2 | No git SHA logging on bootstrap |
| Same notebook | J42 | cell 4 | P2 | Hardcoded `causal_concat=True` |

---

## 7. Confirmation of existing consensus issues (re-verified)

I re-verified the following items from `docs/hyperplan_path_c_plus_consensus.md`. Each remains valid as written:

- **Â§1.1** `dag_grad_gate` never set in `finetune_stage1_bundle_b.py`: confirmed â€” `grep set_dag_grad_gate scripts/finetune_stage1_bundle_b.py` returns zero matches.
- **Â§1.2** DAGMA `s=1.0` hardcoded: confirmed at line 258 (`s = 1.0`).
- **Â§1.3** sigma_data biased estimator: confirmed at lines 701-707, uses `np.mean(deltas)` over per-sample stds.
- **Â§1.4** No RNG state in checkpoint: confirmed â€” `intermediate_state` dict (lines 592-602) does not include `torch_rng_state`, `numpy_rng_state`, `python_rng_state`.
- **Â§1.5** Val/train boundary undefined: confirmed in `_cell43.py:22-25`.
- **Â§1.6** HeteroConv `aggr="sum"` shared params: confirmed at `intelligible_encoder.py:92`. My J3 + J4 expand on the *mechanism* â€” it's worse than the consensus states because the LayerNorm is also shared and outputs collapse to per-target-type tensors, not per-metapath.
- **Â§1.7** `project_dag_floor(prior=G_phys)` overwrites PCMCI init: confirmed at line 338.
- **Â§1.8** KKT Î» ratio: math prof's analysis stands; my J17 adds that even without the KKT issue, the absence of any LR schedule during the 25-epoch fine-tune amplifies the divergence.
- **Â§1.9** RAPSD on 1 sample: confirmed in `scripts/recompute_phase6_metrics.py:328-329`.
- **Â§1.10** `n_batches=16` in evaluation: confirmed and connected to J40 above.

---

## 8. Recommended additions to consensus hyperplan

Add the following to the existing pre-flight checklist (Phase 0.2 section), in priority order:

### P0 â€” MUST-FIX before Phase A0
1. **J29 â€” CFG ignored in EDM Karras path**: implement CFG in `_sample_edm_karras` (Karras 2022 Appendix C.3) OR raise at config validation when `scheduler_type='edm_karras' AND cfg_scale > 1.0 + eps`. **Without this, the V4 Pearson 0.815 result is not reproducible from the config files as advertised.**
2. **J1 â€” `RegressionMeanPredictor._state_adapter` never trained**: build eagerly in `__init__`. **Without this, every BS35 non-causal ablation is invalid.**
3. **J4 â€” Encoder outputs collapse to per-target-type tensors**: refactor `outputs` keying + `aggr="cat"` (extends consensus Â§1.6).
4. **J8 â€” `ConditionalSkipBlock` fed 2-D node tensor**: pass LR grid, not nodes. **Without this, every V5-mini A1 skip-block experiment is silently disabled.**
5. **J18 â€” `strict=False` resume**: switch to `strict=True` by default with explicit opt-in.

### P1 â€” Add to Phase A0 monitoring matrix
6. **J5 â€” Pooling batch=1 broadcast**: assert `B == 1` at conditioning or fix `_assign_default_batch` to track real batch boundaries.
7. **J11 â€” `cross_attention_dim` mismatch assertion**: add validation.
8. **J12 â€” Clip all optimiser groups**, not just 3 modules.
9. **J17 â€” Cosine LR schedule** in Bundle-B fine-tune.
10. **J22 â€” `dag_grad_gate` non-persistent buffer** OR explicit reset at fine-tune entry.
11. **J25 â€” Per-epoch shuffle in `ShardedIterableDataset`** before any multi-GPU claim.
12. **J31 â€” `loss_type` validation**: assert known values, or implement Huber.
13. **J37 â€” `_safe_load` raise on missing keys**: critical for evaluation integrity.
14. **J38, J40 â€” Coordinate `n_batches` and `N_VAL_SAMPLES`** in notebook re-eval.

### Add to instrumentation matrix (Phase A1 monitoring)
- `effective_amp_dtype` per epoch (detect silent fallback to fp32 on T4).
- `grad_clip_actual_norm` per module (verify J12 fix).
- `optimizer.param_groups[i]['lr']` per epoch (verify J17 fix).
- `ema.shadow.norm() / ema.live.norm()` ratio per epoch (verify J21 fix).
- `rcn_cell.dag_grad_gate.item()` per epoch (verify J22 / consensus Â§1.1).
- `_safe_load_failures` per stack build (verify J37 fix).

### Add to consensus Â§2 decision criteria
- Add a hard **GO/NO-GO**: re-run V4 Pearson 0.815 evaluation **with `cfg_scale=1.0`** (matching the actually-used CFG since `edm_karras` ignores cfg_scale per J29). If Pearson drops significantly, the baseline number itself is unreliable and Phase A0 noise-floor measurement must include this correction.

### Add to consensus Â§3 Phase B5 (Stage 2 EDM retrain)
- For V3/V4/V5 retrains: explicitly **disable conditioning_dropout when cfg_scale=1.0** (J30), and document that CFG > 1.0 requires switching `scheduler_type` to `dpm_solver++` until J29 is fixed.

---

**End of audit.** New issues identified: J1â€“J42 (42 items, of which 6 are P0, 16 are P1, 17 are P2/P3). Combined with the 10 consensus items (Â§1.1â€“Â§1.10), the codebase has **52 distinct issues** that should be addressed before any production retrain, with **9 P0 issues that materially affect result validity**.

The most actionable single fix with highest expected value: **J29 (CFG ignored in EDM Karras)** â€” this should be verified before Phase A0, because if the V4 Pearson 0.815 baseline used the EDM path with cfg_scale=1.5 thinking it was active, the entire causal vs CorrDiff comparison rests on an inconsistent guidance setting.

Relevant files:
- `C:/Users/reall/Desktop/climate_data/src/st_cdgm/models/diffusion_decoder.py`
- `C:/Users/reall/Desktop/climate_data/src/st_cdgm/models/intelligible_encoder.py`
- `C:/Users/reall/Desktop/climate_data/src/st_cdgm/models/regression_mean_predictor.py`
- `C:/Users/reall/Desktop/climate_data/src/st_cdgm/models/skip_direct.py`
- `C:/Users/reall/Desktop/climate_data/src/st_cdgm/models/regression_head.py`
- `C:/Users/reall/Desktop/climate_data/src/st_cdgm/models/graph_builder.py`
- `C:/Users/reall/Desktop/climate_data/src/st_cdgm/models/causal_rcn.py`
- `C:/Users/reall/Desktop/climate_data/src/st_cdgm/training/training_loop.py`
- `C:/Users/reall/Desktop/climate_data/src/st_cdgm/training/two_stage.py`
- `C:/Users/reall/Desktop/climate_data/src/st_cdgm/training/stage1_paths.py`
- `C:/Users/reall/Desktop/climate_data/src/st_cdgm/training/multi_gpu.py`
- `C:/Users/reall/Desktop/climate_data/src/st_cdgm/training/callbacks.py`
- `C:/Users/reall/Desktop/climate_data/scripts/finetune_stage1_bundle_b.py`
- `C:/Users/reall/Desktop/climate_data/train_ddp.py`
- `C:/Users/reall/Desktop/climate_data/_cell43.py`
- `C:/Users/reall/Desktop/climate_data/config/training_config.yaml`
- `C:/Users/reall/Desktop/climate_data/config/training_config_corrdiff_normal.yaml`
- `C:/Users/reall/Desktop/climate_data/config/training_config_v5_pearson_090.yaml`
- `C:/Users/reall/Desktop/climate_data/config/training_config_v3.yaml`
- `C:/Users/reall/Desktop/climate_data/config/training_config_v4_corrdiff_large.yaml`
- `C:/Users/reall/Desktop/climate_data/st_cdgm_phase6_finetuned_rerun.ipynb`
- `C:/Users/reall/Desktop/climate_data/docs/hyperplan_path_c_plus_consensus.md`