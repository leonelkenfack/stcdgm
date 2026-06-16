"""Add detailed run-time logs to seed42-v2 notebook (cells 10/11/12).

Goal : operator can see at a glance whether the v2 training behaves normally :
- cache stats before/after Mardani fix (mu_HR really zeroed ? delta unchanged ?)
- per-epoch loss decomposition + DAG-frozen drift + EMA divergence + ETA + verdict
- eval-time per-batch pred/target stats + sanity-range checks against expected
"""
from __future__ import annotations
import json
from pathlib import Path

NB = Path(r"c:/Users/reall/Desktop/climate_data/path_c_plus/scripts/st_cdgm_seed42_eval.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))


def cell(cell_id: str) -> dict:
    for c in nb["cells"]:
        if c.get("id") == cell_id:
            return c
    raise KeyError(cell_id)


def find_line(src: list[str], needle: str) -> int:
    for i, ln in enumerate(src):
        if needle in ln:
            return i
    raise RuntimeError(f"anchor not found: {needle!r}")


def insert_after(src: list[str], needle: str, block: str) -> None:
    i = find_line(src, needle)
    lines = [l + "\n" for l in block.rstrip("\n").split("\n")]
    src[i + 1 : i + 1] = lines


# =================================================================== Cell 10
c10 = cell("cell-10-fix2-prep")
src10 = c10["source"]

# Anchor 1 : right after the v1 cache is loaded/built (the print "N samples : ...")
insert_after(
    src10,
    "[BS32b v1] N samples",
    '''
# === detailed log : v1 cache stats (before Mardani fix) ===
def _stat(_t, _name):
    _t = _t.float()
    _q = torch.quantile(_t.flatten(), torch.tensor([0.5, 0.95, 0.99]))
    print(f"  {_name:14s} shape={tuple(_t.shape)} mean={_t.mean().item():+.4f} "
          f"std={_t.std().item():.4f} absmax={_t.abs().max().item():.4f} "
          f"p50={_q[0].item():+.4f} p95={_q[1].item():+.4f} p99={_q[2].item():+.4f}")
print("[v1 cache stats]")
_stat(bs32b_v1["mu_HR"],        "mu_HR_v1")
_stat(bs32b_v1["baseline_log"], "baseline_log")
_stat(bs32b_v1["delta_target"], "delta_target")
print(f"  valid_mask     valid pct = {100*bs32b_v1['valid_mask'].float().mean().item():.2f}%")
print()
'''.lstrip("\n"),
)

# Anchor 2 : right after bs32b_v2 cache build success print
insert_after(
    src10,
    '"[Mardani fix] verify : delta_target.abs().max()',
    '''
    # === detailed log : v2 cache stats (after Mardani fix) ===
    print("[v2 cache stats (post-Mardani fix)]")
    _stat(bs32b_v2["mu_HR"],        "mu_HR_v2")
    _stat(bs32b_v2["baseline_log"], "baseline_log")
    _stat(bs32b_v2["delta_target"], "delta_target")
    # Verdict on the fix
    _mu_max = bs32b_v2["mu_HR"].abs().max().item()
    _delta_diff = (bs32b_v2["delta_target"] - bs32b_v1["delta_target"]).abs().max().item()
    print(f"[Mardani fix verdict]")
    print(f"  mu_HR_v2 max abs       = {_mu_max:.2e}  ({'OK (==0)' if _mu_max < 1e-10 else 'WARN (should be 0)'})")
    print(f"  delta_target unchanged = {_delta_diff:.2e}  ({'OK (==0)' if _delta_diff < 1e-10 else 'WARN (delta was modified)'})")
    print()
'''.rstrip("\n"),
)

# Anchor 3 : right after diffusion_v2 params print
insert_after(
    src10,
    "[v2 stack] diffusion_v2 :",
    '''
# === detailed log : diffusion_v2 architecture sanity ===
_unet_in = _diff_v2_in = None
try:
    _u = getattr(diffusion_v2, "unet", None) or getattr(getattr(diffusion_v2, "_orig_mod", diffusion_v2), "unet", None)
    if _u is not None and hasattr(_u, "config"):
        _unet_in = int(_u.config.in_channels)
except Exception: pass
print(f"[v2 stack] UNet in_channels = {_unet_in}  (causal_concat=True -> expected 3 = [delta_noisy, mu_HR, baseline_log])")
print(f"[v2 stack] sigma_data={diffusion_v2.edm_config.sigma_data:.5f} | "
      f"sigma_min={diffusion_v2.edm_config.sigma_min:.5f} | "
      f"sigma_max={diffusion_v2.edm_config.sigma_max:.3f}")
print(f"[v2 stack] EDM : P_mean={diffusion_v2.edm_config.P_mean} P_std={diffusion_v2.edm_config.P_std} rho={diffusion_v2.edm_config.rho}")
'''.lstrip("\n"),
)

# Anchor 4 : right after cached_dataloader_v2 ready print, BEFORE "Ready for Cell 11"
insert_after(
    src10,
    "cached_dataloader_v2 ready",
    '''
# === detailed log : batch-1 sanity forward pass ===
print("\\n[batch-1 sanity] pulling one batch from cached_dataloader_v2 ...")
_sanity_batch = next(iter(cached_dataloader_v2))
print(f"  batch shapes : mu_HR={tuple(_sanity_batch['mu_HR'].shape)}  "
      f"baseline_log={tuple(_sanity_batch['baseline_log'].shape)}  "
      f"delta_target={tuple(_sanity_batch['delta_target'].shape)}")
_mu_batch_max = _sanity_batch["mu_HR"].abs().max().item()
print(f"  mu_HR in batch absmax = {_mu_batch_max:.2e}  ({'OK (Mardani fix active)' if _mu_batch_max < 1e-10 else 'WARN (mu_HR NOT zero in dataloader!)'})")
print()
'''.lstrip("\n"),
)


# =================================================================== Cell 11
c11 = cell("cell-11-fix2-train")
src11 = c11["source"]

# Anchor 1 : right before "S2_V2_EPOCHS = int(V2_CONFIG[...])"
insert_after(
    src11,
    'print(f"\\n[fresh] no checkpoint at {ck_v2_path}")',
    '''
# === detailed log : pre-training summary ===
print()
print("[pre-training summary]")
print(f"  Total epochs              : {V2_CONFIG['epochs_max']}")
print(f"  Resume from               : epoch {s2_v2_from}")
print(f"  Remaining                 : {V2_CONFIG['epochs_max'] - s2_v2_from}")
print(f"  Batches per epoch         : {len(cached_dataloader_v2)}")
print(f"  Batch size                : {V2_CONFIG['batch_size']}")
print(f"  Optimizer                 : AdamW lr={V2_CONFIG['lr']} wd={V2_CONFIG['weight_decay']}")
print(f"  EMA decay                 : {EMA_DECAY_V2}")
print(f"  lambda_contrastive_dag    : {V2_CONFIG['lambda_contrastive_dag']}")
print(f"  conditioning_dropout_prob : {V2_CONFIG['conditioning_dropout_prob']}  (no-op when mu_HR=0)")
print(f"  use_amp                   : {bool(CONFIG.training.get('use_amp', True))}")
print(f"  grad clip                 : {CONFIG.training.gradient_clipping}")
# A_dag frozen reference (Stage 1 must not drift)
_adag_frozen_init = rcn_cell.A_dag.detach().cpu().clone()
_adag_frozen_norm0 = float(torch.linalg.norm(_adag_frozen_init).item())
print(f"  A_dag frozen norm (init)  : {_adag_frozen_norm0:.6f}  (must stay constant)")
# EMA divergence reference (must be 0 at start)
def _ema_div(_ema, _live):
    _n, _d = 0.0, 0.0
    for _pe, _pl in zip(_ema.parameters(), _live.parameters()):
        _n += float((_pe - _pl).norm().item()) ** 2
        _d += float(_pl.norm().item()) ** 2
    return (_n ** 0.5) / max(_d ** 0.5, 1e-12)
print(f"  EMA divergence (init)     : {_ema_div(ema_diffusion_v2, diffusion_v2):.2e}  (must grow over epochs)")
print()
print("[expected normal ranges] (per Path C+ / V5-mini calibration)")
print("  loss_diff      : starts ~1.0, descends to ~0.1-0.3 by epoch 200")
print("  contrastive_dag: ~0.01-0.1 (small contribution)")
print("  dag_sensitivity: > 0 (DAG used) and ideally < 1.0 (avoid copy-mode)")
print("  epoch time     : ~7-9 min on A100")
print("  A_dag drift    : == 0 (Stage 1 frozen)")
print()
# Loss-explosion safeguard
_LOSS_EXPLODE_THR = 50.0  # if loss > 50, something is very wrong -> abort
_NAN_INF_ABORT = True
'''.lstrip("\n"),
)

# Anchor 2 : replace the existing one-line per-epoch print with a richer block
insert_after(
    src11,
    '          f"{_dt:.1f}s")',
    '''
    # === detailed log : per-epoch verdict ===
    _ld = float(s2_metrics["loss_diff"])
    _lc = float(s2_metrics.get("loss_contrastive_dag", 0.0))
    _ds = float(s2_metrics.get("dag_sensitivity", 0.0))
    # NaN/Inf safety
    if _ld != _ld or _ld == float("inf"):
        print(f"     [STOP] loss is NaN/Inf at epoch {s2v2_epoch+1} -- aborting training")
        if _NAN_INF_ABORT: raise RuntimeError(f"Loss NaN/Inf at epoch {s2v2_epoch+1}")
    if _ld > _LOSS_EXPLODE_THR:
        print(f"     [WARN] loss_diff={_ld:.2f} > {_LOSS_EXPLODE_THR} : possible explosion")
    # Running stats (last 5 epochs)
    _N_RUN = 5
    if len(history_v2["loss_diff_train"]) >= 2:
        _recent = history_v2["loss_diff_train"][-_N_RUN:]
        _mean_recent = sum(_recent) / len(_recent)
        _trend = "DOWN" if _recent[-1] < _recent[0] else "UP  "
        print(f"     loss_diff last-{len(_recent)}-mean = {_mean_recent:.5f}  trend={_trend}  "
              f"contrast last={_lc:.5f}  dag_sens last={_ds:.4f}")
    # A_dag frozen drift check
    _adag_now = rcn_cell.A_dag.detach().cpu()
    _adag_drift = float((_adag_now - _adag_frozen_init).abs().max().item())
    _adag_norm_now = float(torch.linalg.norm(_adag_now).item())
    if _adag_drift > 1e-10:
        print(f"     [WARN] A_dag drift = {_adag_drift:.2e} | norm now={_adag_norm_now:.6f} init={_adag_frozen_norm0:.6f}")
    # EMA divergence
    _emadv = _ema_div(ema_diffusion_v2, diffusion_v2)
    # ETA
    _avg_t = sum(history_v2["epoch_time"]) / max(len(history_v2["epoch_time"]), 1)
    _eta_h = _avg_t * (S2_V2_EPOCHS - (s2v2_epoch + 1)) / 3600.0
    print(f"     EMA divergence={_emadv:.4e}  |  A_dag norm={_adag_norm_now:.6f} (drift={_adag_drift:.1e})  |  ETA {_eta_h:.1f}h")
    # Verdict every 10 epochs
    if (s2v2_epoch + 1) % 10 == 0:
        _ok = True
        if _ld > _LOSS_EXPLODE_THR: _ok = False
        if _adag_drift > 1e-6: _ok = False
        if _emadv < 1e-6 and (s2v2_epoch + 1) > 5: _ok = False  # EMA should be moving
        print(f"     [verdict @ ep{s2v2_epoch+1}] {'[OK]' if _ok else '[WARN]'}  "
              f"  loss={_ld:.4f} | adag_drift={_adag_drift:.1e} | ema_div={_emadv:.2e}")
'''.rstrip("\n"),
)


# =================================================================== Cell 12
c12 = cell("cell-12-fix2-eval")
src12 = c12["source"]

# Anchor 1 : right after _load_sd(...) for ema_state_dict
insert_after(
    src12,
    '_load_sd(diffusion_v2, ck_v2.get("ema_state_dict"))',
    '''
# === detailed log : EMA load verification ===
_n_loaded = 0
_ema_sd = ck_v2.get("ema_state_dict") or {}
for _k in _ema_sd: _n_loaded += 1
print(f"[EMA load] keys in saved ema_state_dict : {_n_loaded}")
print(f"[EMA load] EMA decay used               : {EMA_CHOICE}")
print(f"[EMA load] epoch_done                   : {ck_v2.get('epoch_done')}")
if "history" in ck_v2 and ck_v2["history"].get("loss_diff_train"):
    _h = ck_v2["history"]["loss_diff_train"]
    print(f"[EMA load] last train loss              : {_h[-1]:.5f}  (epoch {len(_h)})")
    print(f"[EMA load] best train loss              : {min(_h):.5f}  (epoch {1 + _h.index(min(_h))})")
'''.lstrip("\n"),
)

# Anchor 2 : enrich the per-batch progress print
# Replace existing print line "f\"  [v2 FINAL_VAL] batch {_count}/..." with a richer one
for i, ln in enumerate(src12):
    if "[v2 FINAL_VAL] batch {_count}/{N_BATCHES} | elapsed=" in ln:
        # Replace with detailed multi-line print
        new_block = (
            '                if _count % 4 == 0 or _count == 1:\n'
            '                    _mu_max_b = _mu_HR.abs().max().item() if _mu_HR is not None else 0.0\n'
            '                    _samp_mean = _samp.mean().item(); _samp_std = _samp.std().item()\n'
            '                    _samp_inter_std = _samp.std(dim=0).mean().item()\n'
            '                    _tgt_mean = _tgt.mean().item(); _tgt_std = _tgt.std().item()\n'
            '                    print(f"  [v2 FINAL_VAL] batch {_count}/{N_BATCHES} | elapsed={time.time()-_t0:.0f}s")\n'
            '                    print(f"    mu_HR.absmax    = {_mu_max_b:.2e}  ({\'OK\' if _mu_max_b < 1e-10 else \'WARN-NOT-ZERO\'})")\n'
            '                    print(f"    pred K={K}      mean={_samp_mean:+.4f} std={_samp_std:.4f}  inter-K std={_samp_inter_std:.4f}")\n'
            '                    print(f"    target          mean={_tgt_mean:+.4f} std={_tgt_std:.4f}")\n'
        )
        src12[i] = new_block
        # Skip the next line if it was the original closing — already handled
        break

# Anchor 3 : append final-eval sanity-range verdict after FINAL_VAL save
insert_after(
    src12,
    'f"SSR={_ssr_v2:.3f} | F1@p99={_f1_v2.get(',
    '''
    # === detailed log : sanity-range verdicts ===
    print()
    print("[v2 eval sanity ranges] (vs noncausal v4 reference)")
    _NC_RMSE, _NC_PEAR, _NC_F1P99 = 0.1243, 0.8344, 0.5123
    _verdicts = []
    _verdicts.append(("RMSE",       _rmse_v2,      f"want < {_NC_RMSE:.4f} (noncausal)", _rmse_v2 < _NC_RMSE))
    _verdicts.append(("Pearson_PS", _corr_per_sample_v2, f"want > {_NC_PEAR:.4f}",       _corr_per_sample_v2 > _NC_PEAR))
    _verdicts.append(("F1@p99",     _f1_v2.get("p99", float("nan")), f"want > {_NC_F1P99:.4f}", _f1_v2.get("p99", 0) > _NC_F1P99))
    _verdicts.append(("SSR",        _ssr_v2,       "want in [0.6, 1.4] (calibration)",  0.6 <= _ssr_v2 <= 1.4))
    _verdicts.append(("RAPSD",      _rapsd_v2,     "want < 0.20 (spectrum match)",      _rapsd_v2 < 0.20))
    for _name, _val, _txt, _ok in _verdicts:
        _tag = "[OK]" if _ok else "[BELOW]"
        print(f"  {_name:12s} {_val:+.5f}  {_txt}  {_tag}")
    _n_ok = sum(int(_ok) for *_, _ok in _verdicts)
    print(f"\\n  Overall : {_n_ok}/{len(_verdicts)} metrics meet/exceed noncausal v4 baseline")
'''.rstrip("\n"),
)


# =================================================================== write back
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("Logs added to cells 10, 11, 12.")
