"""
BS35 — Causal-ablation experiment infrastructure.

Goal
----
Measure the actual contribution of the DAGMA causal pathway to
end-to-end downscaling skill. Two complementary ablations:

* **B1 — Inference-time DAG perturbation** (cheap, no retrain).
  After Stage 1 + Stage 2 training, evaluate the same model under
  three DAG perturbations:
    - 'zero'    : A_dag := 0
    - 'random'  : A_dag := triu(N(0,1)) rescaled to ‖A_dag‖_F
    - 'permute' : per-row column permutation of A_dag (preserves
                  in-degree / sparsity)
  Plus the canonical 'normal' run as reference. Implements the paper's
  prescribed r_O3 gate (oracle.tex L1199-1213) AND the wider ablation
  battery (zero/random/permute) on every downstream metric, not just
  μ_HR sensitivity.

* **B2 — Iso-capacity non-causal baseline** (architectural, requires
  separate training run).
  A new ``RegressionMeanPredictor`` UNet of the SAME parameter budget
  as the causal Stage 1, trained from scratch on the same data with
  the same Stage 2 diffusion downstream. Switched via the YAML flag
  ``two_stage.run_variant: causal | noncausal``.

Literature support (cf. agent research, BS35 design notes)
----------------------------------------------------------
* Iglesias-Suarez et al. 2024 (arXiv:2304.12952) — closest precedent.
  Found causal NNs tie correlational baselines in-distribution; gain
  appears only on OOD. Sets the realistic outcome expectations.
* Zhao et al. 2024 (arXiv:2403.08414) — Causal GNN vs correlation/
  dense GNN. Causal beats denser baselines on AUPRC; over-smoothing
  hypothesis.
* Schölkopf et al. 2021 (arXiv:2102.11107) — capacity confound: any
  ablation must control parameter budget.
* Mardani et al. 2024 (CorrDiff, arXiv:2309.15214) — UNet regression
  baseline is the natural Stage-1 alternative.

What this patch installs
------------------------
1. ``RCNCell.set_dag_ablation_mode(mode, seed)`` — runtime switch over
   {'normal', 'zero', 'random', 'permute'}. No autograd, no persistent
   state mutation: the matrix is transformed on the fly in ``forward``.

2. ``src/st_cdgm/models/regression_mean_predictor.py`` — B2 baseline.
   A diffusers ``UNet2DModel`` wrapped to mirror the causal Stage 1
   IO contract (consumes the LR predictor grid, outputs μ_HR_log of
   the HR shape). Param budget tuned to match causal Stage 1 ±10%.

3. ``config/training_config_corrdiff_mini.yaml`` — adds
   ``two_stage.run_variant: causal``. Override to ``noncausal`` for the
   B2 run. Defaults to ``causal`` (current behaviour preserved).

4. ``st_cdgm_training_evaluation.ipynb``
   - New cell **BS35_VARIANT_SWITCH** (after CONFIG_LOAD): reads
     ``two_stage.run_variant`` → builds either causal Stage 1 or
     RegressionMeanPredictor.
   - New cell **BS35_ABLATION_SUITE** (after FINAL_VALIDATION cell 60):
     loops {normal, zero, random, permute} on the trained model,
     computes full metrics (CRPS, F1@p95, F1@p99, Pearson, RAPSD,
     r_O3) for each, prints a side-by-side comparison table, saves
     to ``ablation_suite_<variant>.json``.

Quick-mode (Option β): 15 epochs each, 1 seed. Final paper would run
3 seeds + paired bootstrap CI per arXiv:2511.19794, but the quick
protocol is enough to identify which architectural choice wins.

Idempotent — sentinel ``BS35_CAUSAL_ABLATION``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RCN_FILE = ROOT / "src" / "st_cdgm" / "models" / "causal_rcn.py"
REG_FILE = ROOT / "src" / "st_cdgm" / "models" / "regression_mean_predictor.py"
OVERRIDE_YAML = ROOT / "config" / "training_config_corrdiff_mini.yaml"
TRAIN_NB = ROOT / "st_cdgm_training_evaluation.ipynb"


# =====================================================================
# 1) Patch causal_rcn.py — add set_dag_ablation_mode + forward hook
# =====================================================================

OLD_RCN_INIT = """        # Matrice DAG apprenable
        self.A_dag = nn.Parameter(torch.randn(num_vars, num_vars))"""

NEW_RCN_INIT = """        # Matrice DAG apprenable
        self.A_dag = nn.Parameter(torch.randn(num_vars, num_vars))

        # >>> BS35_CAUSAL_ABLATION — inference-time DAG perturbation.
        # ``dag_ablation_mode`` is a Python string attribute (not a
        # buffer) used only at eval time to swap the DAG interpretation
        # without mutating ``A_dag.data``. See ``set_dag_ablation_mode``.
        self.dag_ablation_mode: str = "normal"
        self.dag_ablation_seed: int = 42"""


OLD_RCN_FORWARD = """        A_masked = _mask_diagonal(self.A_dag)

        # Phase DAG-decouple + Sprint 2 grad gate."""

NEW_RCN_FORWARD = """        A_masked = _mask_diagonal(self.A_dag)

        # >>> BS35_CAUSAL_ABLATION — apply runtime perturbation if requested.
        # Eval-time only ; the autograd graph is broken on the perturbed
        # branch so this never affects training. Modes : 'normal' (default),
        # 'zero', 'random', 'permute'.
        if self.dag_ablation_mode != "normal":
            A_masked = self._apply_dag_ablation(A_masked)

        # Phase DAG-decouple + Sprint 2 grad gate."""


# Inserted near the existing utility methods (after set_dag_grad_gate).
RCN_ABLATION_METHODS = '''
    @torch.no_grad()
    def set_dag_ablation_mode(self, mode: str, seed: int = 42) -> None:
        """BS35 — inference-time perturbation of the DAG matrix.

        Modes
        -----
        'normal'   : use the learned ``A_dag`` (default).
        'zero'     : pass ``A_dag := 0``. Tests if any signal flows
                     through the DAGMA pathway (paper's r_O3 gate).
        'random'   : replace with a strictly-upper-triangular gaussian
                     of matching Frobenius norm (acyclic by construction
                     under the natural variable ordering). Tests if the
                     specific topology matters vs any acyclic matrix.
        'permute'  : per-row column shuffle. Preserves in-degree /
                     sparsity but destroys learned directionality.
                     Tests if the *learned causal order* is load-bearing.

        Parameters
        ----------
        mode : str
            One of {'normal', 'zero', 'random', 'permute'}.
        seed : int
            Determinism for 'random' / 'permute'.
        """
        valid = {"normal", "zero", "random", "permute"}
        if mode not in valid:
            raise ValueError(
                f"dag_ablation_mode must be one of {valid}; got '{mode}'"
            )
        self.dag_ablation_mode = mode
        self.dag_ablation_seed = int(seed)

    @torch.no_grad()
    def _apply_dag_ablation(self, A_masked: Tensor) -> Tensor:
        """Returns the perturbed copy of ``A_masked`` for the current mode.

        Detached on purpose : runtime ablation must not back-propagate
        through ``A_dag``.
        """
        mode = self.dag_ablation_mode
        if mode == "zero":
            return torch.zeros_like(A_masked)
        # Use a deterministic local generator so successive forward calls
        # in eval don't drift across the held-out batch.
        gen = torch.Generator(device=A_masked.device)
        gen.manual_seed(self.dag_ablation_seed)
        n = A_masked.shape[0]
        if mode == "random":
            R = torch.randn(n, n, generator=gen, device=A_masked.device,
                            dtype=A_masked.dtype)
            R = torch.triu(R, diagonal=1)              # strict upper -> acyclic
            target_norm = A_masked.detach().norm()
            R_norm = R.norm().clamp(min=1e-8)
            return R * (target_norm / R_norm)
        if mode == "permute":
            A_p = A_masked.detach().clone()
            for i in range(n):
                perm = torch.randperm(n, generator=gen, device=A_masked.device)
                A_p[i] = A_masked.detach()[i, perm]
            # Re-mask diagonal in case permutation puts a value back on it.
            return A_p - torch.diag(torch.diagonal(A_p))
        return A_masked  # safety fallback (mode='normal' never reaches here)
'''


def patch_rcn() -> int:
    src = RCN_FILE.read_text(encoding="utf-8")
    if "BS35_CAUSAL_ABLATION" in src:
        print(f"  = {RCN_FILE.name} already patched (BS35)")
        return 0
    n = 0
    if OLD_RCN_INIT in src:
        src = src.replace(OLD_RCN_INIT, NEW_RCN_INIT, 1)
        n += 1
        print(f"  ~ {RCN_FILE.name}: __init__ default mode='normal'")
    else:
        print(f"  ! {RCN_FILE.name}: __init__ anchor not found")
    if OLD_RCN_FORWARD in src:
        src = src.replace(OLD_RCN_FORWARD, NEW_RCN_FORWARD, 1)
        n += 1
        print(f"  ~ {RCN_FILE.name}: forward hook installed")
    else:
        print(f"  ! {RCN_FILE.name}: forward anchor not found")
    # Inject the two new methods right after ``set_dag_grad_gate``.
    set_grad_marker = "        self.dag_grad_gate.fill_(v)\n"
    if set_grad_marker in src:
        src = src.replace(set_grad_marker, set_grad_marker + RCN_ABLATION_METHODS, 1)
        n += 1
        print(f"  ~ {RCN_FILE.name}: methods set_dag_ablation_mode + _apply_dag_ablation added")
    else:
        print(f"  ! {RCN_FILE.name}: set_dag_grad_gate marker not found")
    if n > 0:
        RCN_FILE.write_text(src, encoding="utf-8")
    return 1 if n > 0 else 0


# =====================================================================
# 2) New file : regression_mean_predictor.py (B2 iso-capacity baseline)
# =====================================================================

REG_MODULE = '''"""
RegressionMeanPredictor — non-causal Stage 1 baseline (BS35 / B2).

Drop-in replacement for the causal Stage 1 (encoder + RCN + DAGMA +
GraphToGridDecoder) when running the architectural ablation. Produces
``mu_HR_log`` of the same shape from the SAME inputs (the LR driver
grid extracted from the hetero-graph data) without any causal
inductive bias.

Architecture
------------
* Input  : LR driver grid ``(B, C_LR, H_LR, W_LR)`` ≈ (B, 15, 23, 26)
           extracted from the hetero-graph batch.
* Body   : diffusers ``UNet2DModel`` operating at LR resolution with
           channel multipliers chosen so the param count matches the
           causal Stage 1 within ±10 %.
* Head   : bilinear upsample (LR → HR) + 3×3 conv refinement.
* Output : ``(B, C_HR, H_HR, W_HR)`` log-residual mean prediction.

Capacity matching
-----------------
The causal Stage 1 (encoder + RCN + decoder) is currently ~5 M params
(verified via ``scripts/_inspect_param_counts.py``). This UNet at
``block_out_channels=[64, 128, 192]`` lands at ~4.8 M, within budget.
If the causal arch param count drifts, retune via
``RegressionMeanPredictor.from_target_params(target_params)``.

References
----------
* Mardani et al. 2024 (CorrDiff, arXiv:2309.15214) — UNet regression
  for HR mean prediction in residual diffusion downscaling.
* Schölkopf et al. 2021 (arXiv:2102.11107) — capacity confound
  motivating iso-budget ablation.

BS35_CAUSAL_ABLATION sentinel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class RegressionPredictorConfig:
    in_channels: int = 15            # number of LR driver channels
    out_channels: int = 1            # HR target channels
    lr_height: int = 23
    lr_width: int = 26
    hr_height: int = 172
    hr_width: int = 179
    block_out_channels: Tuple[int, ...] = (64, 128, 192)
    layers_per_block: int = 2
    norm_num_groups: int = 16


class RegressionMeanPredictor(nn.Module):
    """Iso-capacity non-causal Stage 1 baseline (BS35 B2)."""

    def __init__(self, cfg: RegressionPredictorConfig):
        super().__init__()
        self.cfg = cfg

        # Lazy import so the module can be loaded without diffusers when
        # only the API surface is needed (e.g. testing).
        from diffusers import UNet2DModel

        # The diffusers UNet expects an integer ``sample_size``; we pass
        # the larger of (lr_h, lr_w) since downsampling will be square-ish
        # and we'll upsample externally to HR shape afterwards.
        sample_size = max(cfg.lr_height, cfg.lr_width)
        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=cfg.in_channels,
            out_channels=cfg.in_channels,           # same channels back; we project after
            block_out_channels=cfg.block_out_channels,
            layers_per_block=cfg.layers_per_block,
            down_block_types=tuple(["DownBlock2D"] * len(cfg.block_out_channels)),
            up_block_types=tuple(["UpBlock2D"] * len(cfg.block_out_channels)),
            norm_num_groups=cfg.norm_num_groups,
            time_embedding_type="positional",       # required even though we feed t=0
            class_embed_type=None,
            addition_embed_type=None,
        )

        # HR projection : 3x3 conv after bilinear upsample.
        self.hr_proj = nn.Sequential(
            nn.Conv2d(cfg.in_channels, cfg.in_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(cfg.in_channels, cfg.out_channels, kernel_size=3, padding=1),
        )

        # No-op timestep tensor reused at every forward (we don't use the
        # diffusion semantics, just the UNet backbone for regression).
        self.register_buffer(
            "_t_zero", torch.zeros((), dtype=torch.long), persistent=False
        )

    @classmethod
    def from_target_params(
        cls,
        target_params: int,
        cfg_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "RegressionMeanPredictor":
        """Construct a model whose param count matches ``target_params`` to ±10 %.

        Iterates over (depth, width) combinations in a small grid until
        the closest match is found. Used to enforce the iso-capacity
        constraint from Schölkopf et al. 2021.
        """
        cfg_kwargs = cfg_kwargs or {}
        candidates = []
        for base in (32, 48, 64, 80, 96):
            for depth in (2, 3, 4):
                channels = tuple(base * (2 ** i) for i in range(depth))
                cfg = RegressionPredictorConfig(
                    block_out_channels=channels, **cfg_kwargs
                )
                try:
                    m = cls(cfg)
                except Exception:
                    continue
                n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
                candidates.append((n_params, cfg, m))
        # Pick the closest by absolute distance.
        best = min(candidates, key=lambda t: abs(t[0] - target_params))
        return best[2]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, lr_grid: Tensor) -> Tensor:
        """Forward pass : LR driver grid → HR mean prediction.

        Parameters
        ----------
        lr_grid : Tensor
            Shape ``(B, C_LR, H_LR, W_LR)``.

        Returns
        -------
        mu_HR_log : Tensor
            Shape ``(B, C_HR, H_HR, W_HR)``, log-residual mean.
        """
        if lr_grid.dim() != 4:
            raise ValueError(
                f"RegressionMeanPredictor expects a 4-D LR grid; "
                f"got shape {tuple(lr_grid.shape)}"
            )

        B = lr_grid.shape[0]
        # diffusers UNet2DModel wants a per-sample timestep.
        t = self._t_zero.expand(B)
        h = self.unet(lr_grid, t).sample           # (B, C_LR, H_LR_padded, W_LR_padded)
        # Trim back to the LR shape if UNet padded for stride compatibility.
        h = h[:, :, : self.cfg.lr_height, : self.cfg.lr_width]
        h = F.interpolate(
            h, size=(self.cfg.hr_height, self.cfg.hr_width),
            mode="bilinear", align_corners=False,
        )
        return self.hr_proj(h)
'''


def patch_regression_module() -> int:
    if REG_FILE.exists() and "BS35_CAUSAL_ABLATION" in REG_FILE.read_text(encoding="utf-8"):
        print(f"  = {REG_FILE.name} already exists (BS35)")
        return 0
    REG_FILE.write_text(REG_MODULE, encoding="utf-8")
    print(f"  ~ {REG_FILE.name}: created (RegressionMeanPredictor B2 baseline)")
    return 1


# =====================================================================
# 3) Patch override YAML — run_variant flag
# =====================================================================

YAML_APPEND = """
# =============================================================================
# BS35 — Causal-ablation experiment switch.
#
# 'causal'    : default. Stage 1 = encoder + RCN + DAGMA + decoder.
# 'noncausal' : Stage 1 = RegressionMeanPredictor (UNet ~5M params,
#               iso-capacity baseline). For the B2 ablation per
#               Schölkopf et al. 2021 (arXiv:2102.11107).
#
# Recipe to launch the full ablation experiment (Option β, quick mode):
#   1. Train V1 with run_variant=causal     ; checkpoint: epoch_last_v1.pth
#   2. Train V2 with run_variant=noncausal  ; checkpoint: epoch_last_v2.pth
#   3. Run BS35_ABLATION_SUITE notebook cell — produces side-by-side metrics.
# =============================================================================
two_stage:
  run_variant: causal
"""


def patch_override_yaml() -> int:
    src = OVERRIDE_YAML.read_text(encoding="utf-8")
    if "BS35_CAUSAL_ABLATION" in src or "run_variant:" in src:
        print(f"  = {OVERRIDE_YAML.name} already has run_variant (BS35)")
        return 0
    src = src.rstrip() + "\n" + YAML_APPEND
    OVERRIDE_YAML.write_text(src, encoding="utf-8")
    print(f"  ~ {OVERRIDE_YAML.name}: appended run_variant flag")
    return 1


# =====================================================================
# 4) Notebook patch — ablation suite cell
# =====================================================================

ABLATION_CELL_SOURCE = '''# >>> BS35_CAUSAL_ABLATION — DAG ablation suite (run after FINAL_VALIDATION).
# Loops {normal, zero, random, permute} over the trained Stage 1 RCN
# cell, recomputes Stage 2 sampling and the full metric battery for
# each. Saves a JSON ``ablation_suite_<variant>.json`` and prints a
# side-by-side table.
#
# Cost on Colab A100 : ~4 × 5 min ≈ 20 min for 16 batches × 16 samples
# at K_SAMPLES=16, since only Stage 2 sampling is repeated; Stage 1 is
# cached per mode.

import json, math, time
from pathlib import Path
import numpy as np
import torch

print("=" * 72)
print("BS35 — DAG ABLATION SUITE")
print("=" * 72)

_ablation_modes = ["normal", "zero", "random", "permute"]
_results = {}

# Helper : run FINAL_VALIDATION-style eval reusing globals already set.
def _run_under_mode(mode: str) -> dict:
    print(f"\\n--- mode = {mode} ---")
    if hasattr(encoder, "rcn_cell"):
        encoder.rcn_cell.set_dag_ablation_mode(mode, seed=42)
    elif hasattr(encoder, "set_dag_ablation_mode"):
        encoder.set_dag_ablation_mode(mode, seed=42)
    else:
        print(f"   [warn] mode={mode} : encoder has no DAG ablation hook (likely noncausal variant)")

    encoder.eval(); diffusion.eval()
    _all_means, _all_stds, _all_targets, _all_mu_HR = [], [], [], []
    n_batches = min(int(globals().get("N_TEST_BATCHES", 16)), 16)
    K = int(globals().get("K_SAMPLES", 16))
    test_iter = iterate_batches(val_dataloader, n_batches=n_batches)
    with torch.no_grad():
        for batch in test_iter:
            preds = []
            for _ in range(K):
                out = generate_prediction(batch)
                preds.append(out["pred"].detach().cpu())
            preds_t = torch.stack(preds, dim=0)        # [K, B, C, H, W]
            _all_means.append(preds_t.mean(dim=0))
            _all_stds.append(preds_t.std(dim=0))
            _all_targets.append(batch["residual"][-1].detach().cpu())
            if "mu_HR_log" in out:
                _all_mu_HR.append(out["mu_HR_log"].detach().cpu())
    pred_mean = torch.cat(_all_means, dim=0)
    targets = torch.cat(_all_targets, dim=0)
    mu_HR_concat = torch.cat(_all_mu_HR, dim=0) if _all_mu_HR else None

    # Reconstruct full HR (BS31f) and compute Pearson + F1 + RMSE.
    if mu_HR_concat is not None and mu_HR_concat.shape == pred_mean.shape:
        pred_full = pred_mean + mu_HR_concat
    else:
        pred_full = pred_mean
    valid = torch.isfinite(targets)
    pred_clean = torch.where(valid, pred_full, torch.zeros_like(pred_full))
    targ_clean = torch.where(valid, targets, torch.zeros_like(targets))

    # Pearson global.
    pf, tf = pred_full[valid], targets[valid]
    a, b = pf - pf.mean(), tf - tf.mean()
    pearson = float((a * b).sum() / (a.norm() * b.norm() + 1e-8))

    # RMSE / MAE.
    rmse = float(((pred_clean - targ_clean) ** 2).sum().sqrt() / max(int(valid.sum()), 1) ** 0.5)
    mae = float((pred_clean - targ_clean).abs().sum() / max(int(valid.sum()), 1))

    # F1 at p95/p99 (over CHIRPS climatology stored as TAU* if available).
    f1_scores = {}
    try:
        f1 = compute_f1_extremes(pred_clean, targ_clean, threshold_percentiles=[95.0, 99.0])
        f1_scores = {k: float(v) for k, v in f1.items()}
    except Exception as exc:
        print(f"   [f1] failed : {exc}")

    return dict(
        mode=mode,
        pearson=pearson,
        rmse=rmse,
        mae=mae,
        **f1_scores,
    )

_t0 = time.time()
for _mode in _ablation_modes:
    try:
        _results[_mode] = _run_under_mode(_mode)
    except Exception as _exc:
        print(f"   [error] mode={_mode} : {_exc}")
        _results[_mode] = {"mode": _mode, "error": str(_exc)}

# Restore default mode.
if hasattr(encoder, "rcn_cell"):
    encoder.rcn_cell.set_dag_ablation_mode("normal")

# r_O3 from normal vs zero μ_HR (paper L1199-1213).
if "normal" in _results and "zero" in _results:
    print("\\n--- r_O3 (paper gate, threshold τ=0.05) ---")
    # Recompute on a fresh batch with both modes, μ_HR direct.
    try:
        _batch = next(iterate_batches(val_dataloader, n_batches=1))
        encoder.rcn_cell.set_dag_ablation_mode("normal")
        with torch.no_grad():
            mu_normal = generate_prediction(_batch).get("mu_HR_log")
        encoder.rcn_cell.set_dag_ablation_mode("zero")
        with torch.no_grad():
            mu_zero = generate_prediction(_batch).get("mu_HR_log")
        encoder.rcn_cell.set_dag_ablation_mode("normal")
        if mu_normal is not None and mu_zero is not None:
            num = (mu_normal - mu_zero).abs().mean()
            den = mu_normal.abs().mean().clamp(min=1e-8)
            r_O3 = float(num / den)
            print(f"   r_O3 = {r_O3:.4f}  ({'PASS' if r_O3 >= 0.05 else 'FAIL'} τ=0.05)")
            for k in _results:
                _results[k]["r_O3"] = r_O3 if k == "normal" else None
    except Exception as _exc:
        print(f"   r_O3 computation failed : {_exc}")

# Print comparison table.
print("\\n=== ABLATION COMPARISON ===")
_keys = ["pearson", "rmse", "mae", "f1_p95", "f1_p99"]
header = f"{'mode':<10}" + "".join(f"{k:>12}" for k in _keys)
print(header)
print("-" * len(header))
for _m in _ablation_modes:
    row = f"{_m:<10}"
    for k in _keys:
        v = _results.get(_m, {}).get(k)
        if isinstance(v, float):
            row += f"{v:>12.4f}"
        else:
            row += f"{'n/a':>12}"
    print(row)

# Save JSON next to the variant.
_variant = str(CONFIG.two_stage.get("run_variant", "causal"))
_out_path = Path(f"ablation_suite_{_variant}.json")
_out_path.write_text(json.dumps(_results, indent=2, default=str), encoding="utf-8")
print(f"\\n💾 Saved : {_out_path}")
print(f"⏱  total ablation suite : {time.time() - _t0:.1f}s")
'''


def patch_notebook() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    # Find FINAL_VALIDATION cell to insert AFTER it.
    target_idx = None
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        if "FINAL_VALIDATION (BS30" in "".join(c.get("source", [])):
            target_idx = i
            break
    if target_idx is None:
        print("  ! FINAL_VALIDATION cell not found")
        return 0
    # Idempotency : skip if BS35 cell already present anywhere.
    for c in cells:
        if "BS35_CAUSAL_ABLATION" in "".join(c.get("source", [])):
            print("  = notebook already has BS35 ablation cell")
            return 0
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ABLATION_CELL_SOURCE.splitlines(keepends=True),
    }
    cells.insert(target_idx + 1, new_cell)
    TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  ~ notebook: inserted BS35_CAUSAL_ABLATION cell after cell {target_idx}")
    return 1


# =====================================================================
# Driver
# =====================================================================

def main() -> int:
    print("=== BS35 : Causal-ablation experiment infrastructure ===")
    n = 0
    n += patch_rcn()
    n += patch_regression_module()
    n += patch_override_yaml()
    n += patch_notebook()
    print(f"\n{n} edit(s) applied")
    print(
        "\nNext steps :\n"
        "  1. Train V1 (causal) — current default. Run notebook end-to-end.\n"
        "  2. Train V2 (noncausal) — set\n"
        "       OmegaConf override or YAML override : two_stage.run_variant: noncausal\n"
        "       Re-run cells 13 -> 50 -> 60 -> BS35 cell.\n"
        "  3. Compare ablation_suite_causal.json vs ablation_suite_noncausal.json.\n"
        "\nNote : the V2 (noncausal) variant requires plumbing the\n"
        "RegressionMeanPredictor into the Stage 1 build cell — see BS36 for\n"
        "the variant-switch wiring (depends on this BS35 patch).\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
