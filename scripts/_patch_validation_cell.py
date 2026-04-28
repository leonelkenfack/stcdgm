"""
Insère une cellule "FINAL VALIDATION" à la fin du notebook
``st_cdgm_training_evaluation.ipynb`` qui :

1. Recharge le best checkpoint depuis Drive (fallback last)
2. Pour N=16 batches du val_dataloader :
   a. Encoder + RCN forward → conditioning + spatial_projector(H_T, A_dag)
   b. Sample K=4 fois via DPM-Solver++ (eval_num_steps depuis CONFIG)
3. Pour 4 premiers batches : DAG INTERVENTION TEST
   - Sample avec A_dag réel (cond_spatial complet)
   - Sample avec DAG tokens zéroés
   - Compute delta_zero / signal
4. Métriques agrégées :
   - RMSE / MAE (ensemble mean vs target)
   - Spread (ensemble std)
   - F1_p95, F1_p99 (compute_f1_extremes)
   - RAPSD distance (compute_spectrum_distance)
5. Print + save JSON dans CONFIG.checkpoint.save_dir

Idempotent via la sentinelle ``# >>> FINAL_VALIDATION``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "st_cdgm_training_evaluation.ipynb"
SENTINEL = "# >>> FINAL_VALIDATION"

VALIDATION_MD = """\
## 🔬 Validation finale + Intervention DAG

Cellule à exécuter **après** que la boucle d'entraînement (cell 45) soit
terminée. Elle :

1. recharge le `best` checkpoint depuis Drive,
2. génère K=4 samples par batch via DPM-Solver++ (avec `cfg_scale` du YAML),
3. fait l'**intervention test** sur 4 batches : sample avec `A_dag` réel
   vs avec les DAG tokens zéroés. Mesure `delta_zero / signal` — c'est
   l'indicateur définitif que la diffusion est **causalement conditionnée**
   par le DAG (>1% = conditionne, ~0% = ignoré),
4. agrège F1 p95/p99, RAPSD distance, RMSE,
5. sauvegarde un JSON dans `CONFIG.checkpoint.save_dir`.
"""

VALIDATION_CELL = '''\
# >>> FINAL_VALIDATION
# Validation finale post-training : sampling + intervention DAG.
import json
import time
from pathlib import Path
import numpy as np
import torch

from st_cdgm.evaluation import (
    compute_f1_extremes,
    compute_spectrum_distance,
)


# ── 1. Recharger le checkpoint best (fallback last) ─────────────────
_ckpt_dir = Path(CONFIG.checkpoint.get("save_dir", "models"))
_best = _ckpt_dir / "epoch_best.pth"
_last = _ckpt_dir / "epoch_last.pth"
_ckpt_path = _best if _best.exists() else _last
if not _ckpt_path.exists():
    raise FileNotFoundError(
        f"Aucun checkpoint dans {_ckpt_dir}. La boucle d'entraînement a-t-elle "
        f"persisté au moins une époque (cell 45) ?"
    )
print(f"📦 Loading checkpoint: {_ckpt_path}")
_ckpt = torch.load(_ckpt_path, map_location=DEVICE, weights_only=False)
print(f"   epoch={_ckpt.get('epoch')} / {_ckpt.get('epochs_total')}, "
      f"val_loss={_ckpt.get('val_loss')}, best_val_loss={_ckpt.get('best_val_loss')}")

# Recharge state_dicts via le helper de la cell 44.
_persist_load_state_dict(encoder, _ckpt.get("encoder_state_dict"))
_persist_load_state_dict(rcn_cell, _ckpt.get("rcn_cell_state_dict"))
_persist_load_state_dict(diffusion, _ckpt.get("diffusion_state_dict"))
if "spatial_projector" in dir() and spatial_projector is not None:
    _persist_load_state_dict(spatial_projector, _ckpt.get("spatial_projector_state_dict"))
if "hr_ident_head" in dir() and hr_ident_head is not None:
    _persist_load_state_dict(hr_ident_head, _ckpt.get("hr_ident_head_state_dict"))

encoder.eval()
rcn_runner.cell.eval()
diffusion.eval()
if "spatial_projector" in dir() and spatial_projector is not None:
    spatial_projector.eval()


# ── 2. Helpers : sampling + extract conditioning ────────────────────
def _build_conditioning(_batch):
    """Encoder + RCN → conditioning (FiLM) + conditioning_spatial (cross-attn)."""
    _hetero = _batch["hetero"]
    _lr = _batch["lr"]
    if isinstance(_lr, torch.Tensor):
        _lr = _lr.to(DEVICE)
    _H_init = encoder.init_state(_hetero).to(DEVICE)
    _drivers = [_lr[t] for t in range(_lr.shape[0])]
    _seq = rcn_runner.run(_H_init, _drivers, reconstruction_sources=None)
    _H_T = _seq.states[-1]
    _A_dag = _seq.dag_matrices[-1]

    _cond = encoder.project_state_tensor(_H_T).to(DEVICE)

    _sp_core = _eager_core(spatial_projector) if "spatial_projector" in dir() and spatial_projector is not None else None
    _cond_sp_real = None
    _cond_sp_zero = None
    if _sp_core is not None:
        if hasattr(_sp_core, "num_dag_tokens"):
            _cond_sp_real = spatial_projector(_H_T, _A_dag).to(DEVICE)
            _n_dag = int(_sp_core.num_dag_tokens)
            _cond_sp_zero = _cond_sp_real.clone()
            _cond_sp_zero[:, -_n_dag:, :] = 0.0
        else:
            _cond_sp_real = spatial_projector(_H_T).to(DEVICE)
            _cond_sp_zero = torch.zeros_like(_cond_sp_real)
    return _cond, _cond_sp_real, _cond_sp_zero


def _sample_once(_cond, _cond_sp):
    """Un sample DPM-Solver++ avec eval_num_steps + cfg_scale du YAML."""
    _diff_core = _eager_core(diffusion)
    return _diff_core.sample(
        conditioning=_cond,
        conditioning_spatial=_cond_sp,
        num_steps=int(CONFIG.diffusion.get("eval_num_steps", 30)),
        scheduler_type=CONFIG.diffusion.get("scheduler_type", "dpm_solver++"),
        cfg_scale=float(CONFIG.diffusion.get("cfg_scale", 0.0)),
        apply_constraints=True,
    ).residual


# ── 3. Boucle d'évaluation ──────────────────────────────────────────
N_TEST_BATCHES = 16
K_SAMPLES = 4              # ensemble pour CRPS/spread (4 = compromis vitesse/précision)
N_INTERVENTION = 4         # nb de batches pour le DAG intervention test

print(f"\\n🧪 Sampling {N_TEST_BATCHES} batches × {K_SAMPLES} samples")
print(f"   eval_num_steps={CONFIG.diffusion.get('eval_num_steps', 30)}, "
      f"cfg_scale={CONFIG.diffusion.get('cfg_scale', 0.0)}")

_all_samples_mean = []  # ensemble mean per batch
_all_samples_std = []
_all_targets = []
_dag_intervention = []   # delta_zero / signal per batch (premier N_INTERVENTION uniquement)
_t_eval_start = time.time()

with torch.no_grad():
    for _bi, _raw_batch in enumerate(val_dataloader):
        if _bi >= N_TEST_BATCHES:
            break
        _batch = convert_sample_to_batch(_raw_batch, builder, DEVICE)
        _cond, _cond_sp_real, _cond_sp_zero = _build_conditioning(_batch)

        # Cible HR (résiduel — dernier timestep, batché à [1, C, H, W]).
        _target = _batch[CONFIG.training.get("residual_key", "residual")][-1].to(DEVICE)
        if _target.dim() == 3:
            _target = _target.unsqueeze(0)

        # K samples conditionnés sur le vrai A_dag → ensemble.
        _samples_k = []
        for _k in range(K_SAMPLES):
            _s = _sample_once(_cond, _cond_sp_real)
            _samples_k.append(_s)
        _samples_k = torch.stack(_samples_k, dim=0)   # [K, B, C, H, W]
        _all_samples_mean.append(_samples_k.mean(dim=0))
        _all_samples_std.append(_samples_k.std(dim=0))
        _all_targets.append(_target)

        # Intervention DAG sur les premiers N_INTERVENTION batches.
        if _bi < N_INTERVENTION and _cond_sp_zero is not None:
            _s_real = _sample_once(_cond, _cond_sp_real)
            _s_zero = _sample_once(_cond, _cond_sp_zero)
            _delta = (_s_real - _s_zero).abs().mean().item()
            _signal = _s_real.abs().mean().item()
            _ratio = _delta / max(_signal, 1e-8)
            _dag_intervention.append(_ratio)
            print(f"   batch {_bi+1:2d} | delta_zero/signal = {_ratio*100:6.2f}%")
        else:
            print(f"   batch {_bi+1:2d} | sampled K={K_SAMPLES}")

_eval_time = time.time() - _t_eval_start

# ── 4. Métriques agrégées ───────────────────────────────────────────
_pred_mean = torch.cat(_all_samples_mean, dim=0).cpu()    # [N, C, H, W]
_pred_std = torch.cat(_all_samples_std, dim=0).cpu()
_targets = torch.cat(_all_targets, dim=0).cpu()

# Mask NaN du target (océan, etc.) avant les métriques pixel-wise.
_valid = torch.isfinite(_targets)
_pred_clean = torch.where(_valid, _pred_mean, torch.zeros_like(_pred_mean))
_targ_clean = torch.where(_valid, _targets, torch.zeros_like(_targets))

# RMSE / MAE sur pixels valides.
_diff_sq = ((_pred_clean - _targ_clean) ** 2)[_valid]
_rmse = float(_diff_sq.mean().sqrt().item()) if _valid.any() else float("nan")
_mae = float((_pred_clean - _targ_clean).abs()[_valid].mean().item()) if _valid.any() else float("nan")
_spread = float(_pred_std[_valid].mean().item()) if _valid.any() else float("nan")

# F1 extrêmes via la lib existante.
_f1 = {}
try:
    _f1 = compute_f1_extremes(_pred_clean, _targ_clean, threshold_percentiles=[95.0, 99.0])
except Exception as _e:
    print(f"⚠️  F1 extremes a échoué : {_e}")

# RAPSD distance (sample [0, 0]).
_rapsd_d = None
try:
    _rapsd_d = float(compute_spectrum_distance(_pred_clean[0], _targ_clean[0]))
except Exception as _e:
    print(f"⚠️  RAPSD distance a échoué : {_e}")

# DAG intervention.
_dag_avg = float(np.mean(_dag_intervention)) if _dag_intervention else None
_dag_pct = (_dag_avg * 100.0) if _dag_avg is not None else None

# ── 5. Print + JSON ─────────────────────────────────────────────────
print("\\n" + "=" * 72)
print("📊 FINAL VALIDATION METRICS")
print("=" * 72)
print(f"  Checkpoint            : {_ckpt_path.name}  (epoch {_ckpt.get('epoch')})")
print(f"  Test batches          : {len(_all_targets)}")
print(f"  Samples/batch         : {K_SAMPLES}")
print(f"  Eval time             : {_eval_time:.1f}s ({_eval_time/max(1,len(_all_targets)):.1f}s/batch)")
print()
print(f"  RMSE (ensemble mean)  : {_rmse:.6f}")
print(f"  MAE                   : {_mae:.6f}")
print(f"  Spread (ensemble std) : {_spread:.6f}")
for _k, _v in _f1.items():
    print(f"  {_k:<22}: {_v:.4f}")
if _rapsd_d is not None:
    print(f"  RAPSD distance        : {_rapsd_d:.6f}")
print()
if _dag_pct is not None:
    print(f"  🧠 INTERVENTION TEST (DAG conditioning):")
    print(f"     delta_zero / signal = {_dag_pct:.3f}%   (sur {len(_dag_intervention)} batches)")
    if _dag_avg < 0.001:
        print(f"     ⚠️  DAG IGNORÉ — UNet n'utilise pas les DAG tokens (ratio < 0.1%)")
    elif _dag_avg < 0.01:
        print(f"     ⚠️  DAG WEAKLY CONDITIONS (0.1% < ratio < 1%)")
    else:
        print(f"     ✅ DAG CONDITIONS la diffusion (ratio ≥ 1%)")
else:
    print(f"  ⚠️  Intervention test non exécuté (spatial_projector absent ou pas de DAG tokens)")

# Sauvegarde JSON pour comparaison entre runs.
_metrics = {
    "checkpoint": str(_ckpt_path),
    "epoch": _ckpt.get("epoch"),
    "epochs_total": _ckpt.get("epochs_total"),
    "best_val_loss": _ckpt.get("best_val_loss"),
    "n_test_batches": len(_all_targets),
    "k_samples": K_SAMPLES,
    "eval_time_s": _eval_time,
    "rmse": _rmse,
    "mae": _mae,
    "spread_mean": _spread,
    "f1_extremes": _f1,
    "rapsd_distance": _rapsd_d,
    "dag_intervention": {
        "delta_zero_signal_ratio_avg": _dag_avg,
        "delta_zero_signal_pct": _dag_pct,
        "per_batch": _dag_intervention,
        "verdict": (
            "DAG_CONDITIONS" if (_dag_avg is not None and _dag_avg >= 0.01) else
            "WEAK" if (_dag_avg is not None and _dag_avg >= 0.001) else
            "IGNORED" if _dag_avg is not None else
            "N/A"
        ),
    },
    "config_eval_num_steps": int(CONFIG.diffusion.get("eval_num_steps", 30)),
    "config_cfg_scale": float(CONFIG.diffusion.get("cfg_scale", 0.0)),
    "config_block_out_channels": list(CONFIG.diffusion.unet_kwargs.block_out_channels),
}
_metrics_path = _ckpt_dir / "final_validation_metrics.json"
_metrics_path.write_text(json.dumps(_metrics, indent=2, default=str))
print(f"\\n💾 Métriques sauvegardées : {_metrics_path}")
'''


def _make_md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _make_code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def main() -> int:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # Idempotency : check existing sentinel
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if SENTINEL in src:
            # Réécrire pour synchro contenu (le code peut avoir évolué)
            cells[i]["source"] = VALIDATION_CELL.splitlines(keepends=True)
            cells[i]["outputs"] = []
            cells[i]["execution_count"] = None
            NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"✓ Cell validation existante (idx={i}) — réécrite pour synchro.")
            return 0

    # Append à la fin
    cells.append(_make_md_cell(VALIDATION_MD))
    cells.append(_make_code_cell(VALIDATION_CELL))
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  ↪ 2 cellules ajoutées en fin de notebook (markdown + code, idx={len(cells)-2}, {len(cells)-1})")
    print(f"💾 Saved {NB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
