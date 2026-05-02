"""
Finitions Colab : ajoute aux notebooks tout ce qui reste pour un run
20h end-to-end propre.

Training notebook
-----------------
1. ``# >>> EDM_GPU_SANITY_CHECK`` au début : confirme A100 + bf16 + VRAM.

Validation notebook
-------------------
2. ``# >>> EDM_ABLATION_A_DAG_GATE`` : reload checkpoint, set
   dag_grad_gate=1.0, train 5 epochs courtes, comparer F1_p99.
3. ``# >>> EDM_ABLATION_B_CFG`` : re-évaluer 50 samples avec
   cfg_scale=2.0 sur le même checkpoint, mesurer impact sur sigma_r.
4. ``# >>> EDM_ABLATION_C_NO_CROSSATTN`` : note explicative + skip
   conditionnel (vraie ablation = re-train, hors budget Colab).
5. ``# >>> EDM_RAPSD_PLOTS`` : RAPSD spectrum + scatter publication-ready.
6. ``# >>> EDM_FINAL_SUMMARY`` : agrège shd_report + intervention_report
   + global_summary + thresholds dans un dict + dump JSON unique.

Toutes les insertions sentinellées → idempotent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_NB = ROOT / "st_cdgm_training_evaluation.ipynb"
VAL_NB = ROOT / "st_cdgm_validation_inference.ipynb"


# ---------------------------------------------------------------------
# CELL BLOCKS
# ---------------------------------------------------------------------

GPU_SANITY_CHECK = r'''# >>> EDM_GPU_SANITY_CHECK
# Garde-fou Colab : confirme que le runtime est bien A100 (sm_80+),
# que bf16 est supporté, et que la VRAM disponible est compatible
# avec batch_size=16 micro × 2 accum + UNet [32, 64].
import torch

print("=" * 60)
print("GPU sanity check (Colab)")
print("=" * 60)

if not torch.cuda.is_available():
    print("⚠️  CUDA indisponible — running CPU mode (debug only).")
else:
    _props = torch.cuda.get_device_properties(0)
    _cap = torch.cuda.get_device_capability(0)
    _free, _total = torch.cuda.mem_get_info()
    print(f"  Device : {_props.name}")
    print(f"  Compute capability : sm_{_cap[0]}{_cap[1]}")
    print(f"  VRAM total : {_total / 1e9:.1f} GB")
    print(f"  VRAM free  : {_free / 1e9:.1f} GB")
    print(f"  bf16 supporté (sm_80+) : {_cap[0] >= 8}")

    if _cap[0] < 8:
        print("  ⚠️  sm_<80 — bf16 non disponible, fallback fp16+GradScaler.")
        print("     Training EDM peut overflow en fp16 — recommandation : T4 OK pour smoke, A100 pour grand run.")
    if _total < 15e9:
        print("  ⚠️  VRAM < 15 GB — réduire batch_size ou block_out_channels.")
    elif _total < 40e9:
        print("  ℹ️  VRAM 15-40 GB — config OK pour batch_size=16, possibilité d'augmenter.")
    else:
        print(f"  ✓ A100 80GB détecté — config par défaut OK, marge confortable.")

print("=" * 60)
'''


ABLATION_A_GATE = r'''# >>> EDM_ABLATION_A_DAG_GATE
# Ablation A : ré-active dag_grad_gate=1.0 sur le checkpoint epoch_last,
# train 5 epochs supplémentaires, valide sur 50 samples.
# Objectif : prouver que detach > grad-flow (Trace Trap fermé > ouvert).
#
# /!\ Coût : ~1.5h de training A100 supplémentaire. Skip si on n'a
# pas la marge dans le budget 20h.

ABLATION_A_ENABLED = False  # mettre True manuellement quand tu veux lancer

if ABLATION_A_ENABLED:
    print("🧪 Ablation A : dag_grad_gate=1.0, +5 epochs depuis epoch_last")
    print("   ⚠️  Cette cellule re-entraîne — coût ~1.5h.")

    # 1. Backup du checkpoint actuel (ne PAS écraser)
    import shutil
    from pathlib import Path
    _ck_dir = Path(str(CONFIG.checkpoint.save_dir))
    _ck_main = _ck_dir / "epoch_last.pth"
    _ck_ablA = _ck_dir / "_ablation_A_gate.pth"
    if _ck_ablA.exists():
        print(f"   ✓ ablation A déjà entraînée : {_ck_ablA}")
    elif _ck_main.exists():
        # 2. Override config et re-entraîne
        CONFIG.rcn.dag_grad_gate.enabled = True
        CONFIG.rcn.dag_grad_gate.cold_epochs = 0
        CONFIG.rcn.dag_grad_gate.ramp_epochs = 1
        CONFIG.rcn.dag_grad_gate.max = 1.0
        CONFIG.training.epochs = 5  # juste 5 ep en plus
        print("   → ré-importer training_loop et lancer train_epoch en boucle")
        print("   → puis sauvegarder vers _ablation_A_gate.pth")
        print("   (Cette cellule est un PLACEHOLDER — implémenter le micro-loop si nécessaire)")
    else:
        print(f"   ✗ checkpoint epoch_last absent : {_ck_main}")
else:
    print("ℹ️  Ablation A désactivée (mettre ABLATION_A_ENABLED=True pour lancer).")
'''


ABLATION_B_CFG = r'''# >>> EDM_ABLATION_B_CFG
# Ablation B : compare sigma_r entre cfg_scale=1.0 (default) et 2.0
# sur 50 samples. Pas de re-training, juste re-inférence.
# Objectif : confirmer que CFG > 1 inflate la variance d'ensemble
# (Gemini Axis 5 + reviewer paper).
import numpy as np
import torch

ABLATION_B_ENABLED = True  # rapide (~30 min)
ABLATION_B_N_SAMPLES = 30

if ABLATION_B_ENABLED:
    print(f"🧪 Ablation B : cfg_scale=1.0 vs 2.0 sur {ABLATION_B_N_SAMPLES} samples")

    _cfg_orig = float(CONFIG.diffusion.get("cfg_scale", 1.0))

    def _run_with_cfg(sample, cfg_scale_val):
        batch = convert_sample_to_batch(sample, builder, DEVICE)
        lr_data = batch["lr"].to(DEVICE)
        target_batch, baseline_batch = extract_target_and_baseline_t_mean(batch, DEVICE)
        H_init = encoder.init_state(batch["hetero"]).to(DEVICE)
        drivers = [lr_data[t] for t in range(lr_data.shape[0])]
        seq_out = rcn_runner.run(H_init, drivers, reconstruction_sources=None)
        H_last = seq_out.states[-1]
        cond = encoder.project_state_tensor(H_last).to(DEVICE)

        cond_sp = None
        _sp = globals().get("spatial_projector", None)
        if _sp is not None:
            _sp_base = _sp.module if hasattr(_sp, "module") else _sp
            _sp_base = getattr(_sp_base, "_orig_mod", _sp_base)
            _rcn_b = rcn_runner.cell.module if hasattr(rcn_runner.cell, "module") else rcn_runner.cell
            _rcn_b = getattr(_rcn_b, "_orig_mod", _rcn_b)
            if hasattr(_sp_base, "dag_mlp"):
                _A = _rcn_b.A_dag
                _Am = _A - torch.diag(torch.diagonal(_A))
                cond_sp = _sp(H_last, _Am).to(DEVICE)
            else:
                cond_sp = _sp(H_last).to(DEVICE)

        # 5 ensemble members for sigma_r
        members = []
        for k in range(5):
            gen = torch.Generator(device=DEVICE).manual_seed(20000 + k)
            out = diffusion.sample(
                cond,
                num_steps=int(CONFIG.diffusion.get("eval_num_steps", 18)),
                scheduler_type=CONFIG.diffusion.get("scheduler_type", "edm_karras"),
                apply_constraints=False, baseline=baseline_batch,
                cfg_scale=cfg_scale_val,
                conditioning_spatial=cond_sp, generator=gen,
            )
            members.append(out.t_mean.detach().cpu())
        return members, target_batch.cpu()

    _pipe = build_test_pipeline(TEST_GCMS[0])
    _ds = _pipe.build_sequence_dataset(seq_len=SEQ_LEN, as_torch=True)

    sigma_r_cfg1, sigma_r_cfg2 = [], []
    with torch.no_grad():
        for _i, _s in enumerate(_ds):
            if _i >= ABLATION_B_N_SAMPLES:
                break
            mem1, tgt = _run_with_cfg(_s, 1.0)
            mem2, _ = _run_with_cfg(_s, 2.0)
            stack1 = torch.stack(mem1, dim=0)  # [N, B, C, H, W]
            stack2 = torch.stack(mem2, dim=0)
            target_std = tgt.float().std().item() + 1e-8
            sigma_r_cfg1.append(stack1.float().std(dim=0).mean().item() / target_std)
            sigma_r_cfg2.append(stack2.float().std(dim=0).mean().item() / target_std)

    s1 = np.array(sigma_r_cfg1); s2 = np.array(sigma_r_cfg2)
    print(f"\n📊 Ablation B summary")
    print(f"   sigma_r @ cfg=1.0  :  mean={s1.mean():.3f}  median={np.median(s1):.3f}")
    print(f"   sigma_r @ cfg=2.0  :  mean={s2.mean():.3f}  median={np.median(s2):.3f}")
    print(f"   ratio cfg2/cfg1    :  {s2.mean() / max(s1.mean(), 1e-8):.2f}x")
    if s2.mean() > 2.0 * s1.mean():
        print("   ✓ confirmé : cfg=2.0 explose la dispersion (>2x)")
    else:
        print("   ⚠️  cfg n'inflate pas autant qu'attendu — investigate")

    ablation_B_report = {
        "n_samples": int(len(s1)),
        "sigma_r_cfg1_mean": float(s1.mean()),
        "sigma_r_cfg2_mean": float(s2.mean()),
        "ratio_cfg2_cfg1": float(s2.mean() / max(s1.mean(), 1e-8)),
    }
else:
    print("ℹ️  Ablation B désactivée (ABLATION_B_ENABLED=False).")
    ablation_B_report = None
'''


ABLATION_C_NOTE = r'''# >>> EDM_ABLATION_C_NO_CROSSATTN
# Ablation C : architecture sans cross-attention (full DownBlock2D / UpBlock2D).
# REQUIERT un re-training depuis zéro car la topologie UNet change.
# Estimé ~3-4h GPU + validation ~1h → ~5h total.
#
# Plutôt que de l'exécuter ici (hors budget 20h), on documente la procédure :
#   1. Backup config actuelle
#   2. Modifier YAML : down_block_types=[DownBlock2D, DownBlock2D],
#      up_block_types=[UpBlock2D, UpBlock2D], mid_block_type=UNetMidBlock2D
#   3. Re-train 30 epochs avec scheduler_type=edm_karras
#   4. Comparer F1_p99, sigma_r, delta_zero/signal aux résultats edm-rewrite
#
# Cette ablation valide si la cross-attention est load-bearing pour le
# conditioning DAG. Si F1_p99 baisse de >20% en l'enlevant, la cross-attn
# joue un rôle. Sinon le projector class_embed seul suffit.
print("ℹ️  Ablation C (no cross-attn) : voir docstring de cette cellule.")
print("   → planifier comme Phase 6 (post-Colab) si budget permet.")
'''


RAPSD_PLOTS = r'''# >>> EDM_RAPSD_PLOTS
# Figures publication-ready : RAPSD spectrum (target vs prediction
# sur l'ensemble) + scatter plot prédiction vs target sur événements
# extrêmes (>p95).
import numpy as np
import torch
import matplotlib.pyplot as plt

from st_cdgm.evaluation.evaluation_xai import compute_rapsd_numpy

print("📈 Génération des figures publication-ready...")

# Aggrege les samples_out de la dernière itération de la boucle eval.
# Sécurité : si la boucle eval n'a pas encore tourné, skip.
if "samples_out" not in dir() or "target_batch" not in dir():
    print("⚠️  Eval loop pas encore exécutée — skip RAPSD plots.")
else:
    # 1. RAPSD spectrum (averaged across the ensemble of 5 members)
    pred_stack = torch.stack([s.t_mean for s in samples_out], dim=0)  # [N, B, C, H, W]
    pred_mean_field = pred_stack.mean(dim=0)  # [B, C, H, W]

    # Extract 2D slices
    def _to_2d(t):
        a = np.asarray(t.detach().cpu() if hasattr(t, "detach") else t)
        while a.ndim > 2:
            a = a[0]
        return a

    p_2d = _to_2d(pred_mean_field)
    t_2d = _to_2d(target_batch)

    rapsd_pred = compute_rapsd_numpy(p_2d)
    rapsd_targ = compute_rapsd_numpy(t_2d)

    nbin = min(len(rapsd_pred), len(rapsd_targ))
    freqs = np.arange(nbin)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1 : RAPSD log-log
    axes[0].loglog(freqs[1:], rapsd_targ[1:nbin], "k-", lw=2, label="Target")
    axes[0].loglog(freqs[1:], rapsd_pred[1:nbin], "C0--", lw=2, label="Prediction (ens. mean)")
    axes[0].set_xlabel("Wavenumber (px⁻¹)")
    axes[0].set_ylabel("Spectral power")
    axes[0].set_title("RAPSD — Radial spectrum")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    # Panel 2 : Scatter pred vs target on top-percentile pixels
    p_flat = p_2d.flatten()
    t_flat = t_2d.flatten()
    p95 = np.percentile(t_flat, 95)
    mask = t_flat > p95
    axes[1].scatter(t_flat[mask], p_flat[mask], alpha=0.3, s=8)
    _max = max(t_flat[mask].max(), p_flat[mask].max())
    axes[1].plot([0, _max], [0, _max], "r--", alpha=0.5, label="y=x")
    axes[1].set_xlabel("Target (>p95)")
    axes[1].set_ylabel("Prediction")
    axes[1].set_title(f"Extreme events (top 5%, n={int(mask.sum())})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3 : RAPSD distance per frequency band (diagnostic)
    rel_err = np.abs(rapsd_pred[1:nbin] - rapsd_targ[1:nbin]) / (rapsd_targ[1:nbin] + 1e-12)
    axes[2].semilogx(freqs[1:], rel_err)
    axes[2].set_xlabel("Wavenumber (px⁻¹)")
    axes[2].set_ylabel("Relative error |pred-target|/target")
    axes[2].set_title("RAPSD relative error per frequency")
    axes[2].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    _out_path = RESULTS_DIR / "rapsd_publication.png"
    plt.savefig(_out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"   ✓ Figure sauvée : {_out_path}")
'''


FINAL_SUMMARY = r'''# >>> EDM_FINAL_SUMMARY
# Agrège tous les rapports en un JSON unique pour archivage / paper.
import json as _json
from pathlib import Path

_summary = {
    "checkpoint_path": str(CHECKPOINT_PATH),
    "scheduler_type": str(CONFIG.diffusion.get("scheduler_type", "ddpm")),
    "edm_config": (
        {
            "sigma_data": float(CONFIG.diffusion.edm.sigma_data),
            "sigma_min": float(CONFIG.diffusion.edm.sigma_min),
            "sigma_max": float(CONFIG.diffusion.edm.sigma_max),
            "rho": float(CONFIG.diffusion.edm.rho),
            "P_mean": float(CONFIG.diffusion.edm.P_mean),
            "P_std": float(CONFIG.diffusion.edm.P_std),
        }
        if CONFIG.diffusion.get("scheduler_type") == "edm_karras"
        else None
    ),
    "global_metrics": globals().get("global_summary", {}),
    "shd_report": globals().get("shd_report"),
    "intervention_report": globals().get("intervention_report"),
    "ablation_B_report": globals().get("ablation_B_report"),
    "non_regression_pass": bool(globals().get("non_reg_pass", False)),
}

_out_path = RESULTS_DIR / "edm_final_summary.json"
with open(_out_path, "w", encoding="utf-8") as _f:
    _json.dump(_summary, _f, indent=2, default=str)

print("=" * 70)
print("📋 FINAL SUMMARY (EDM run)")
print("=" * 70)
print(f"  Checkpoint : {_summary['checkpoint_path']}")
print(f"  Scheduler  : {_summary['scheduler_type']}")
if _summary["edm_config"]:
    print(f"  σ_data     : {_summary['edm_config']['sigma_data']:.4f}")
print(f"  Non-regression pass : {_summary['non_regression_pass']}")
if _summary["intervention_report"]:
    _ir = _summary["intervention_report"]
    print(f"  DAG intervention : delta/signal = {_ir.get('delta_signal_ratio_mean'):.4f} "
          f"({'PASS' if _ir.get('passes_o6_threshold') else 'FAIL'})")
if _summary["shd_report"]:
    print(f"  SHD vs prior : total = {_summary['shd_report'].get('total')}")
if _summary["ablation_B_report"]:
    _ab = _summary["ablation_B_report"]
    print(f"  Ablation B (CFG): cfg=1 σ_r={_ab['sigma_r_cfg1_mean']:.3f}, "
          f"cfg=2 σ_r={_ab['sigma_r_cfg2_mean']:.3f}, ratio={_ab['ratio_cfg2_cfg1']:.2f}x")
print(f"\n  💾 JSON dump : {_out_path}")
print("=" * 70)
'''


# ---------------------------------------------------------------------
# patcher
# ---------------------------------------------------------------------

def _make_code_cell(src, cell_id):
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def _find_cell(cells, predicate):
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        if predicate("".join(c.get("source", []))):
            return i
    return None


def patch_training_notebook() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    n_changed = 0

    if _find_cell(cells, lambda s: "EDM_GPU_SANITY_CHECK" in s) is None:
        # Insérer après le bootstrap (cell 1), avant les imports lourds
        boot_idx = _find_cell(cells, lambda s: "COLAB_BOOTSTRAP" in s)
        if boot_idx is not None:
            cells.insert(boot_idx + 1, _make_code_cell(GPU_SANITY_CHECK, "edm_gpu_sanity"))
            n_changed += 1
            print(f"  + training EDM_GPU_SANITY_CHECK inséré en cell {boot_idx + 1}")
    else:
        print("  = training EDM_GPU_SANITY_CHECK déjà présent")

    if n_changed > 0:
        TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return n_changed


def patch_validation_notebook() -> int:
    nb = json.loads(VAL_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    n_changed = 0

    # Anchor : on insère après l'eval principale (for gcm in TEST_GCMS).
    eval_idx = _find_cell(cells, lambda s: "for gcm in TEST_GCMS" in s)
    if eval_idx is None:
        print("  ! eval anchor introuvable")
        return 0

    # Cell ordering : juste après eval, en chaîne :
    #   eval_idx + 1 : RAPSD_PLOTS
    #   eval_idx + 2 : ABLATION_B_CFG
    #   eval_idx + 3 : ABLATION_C_NOTE
    #   eval_idx + 4 : ABLATION_A_GATE (en dernier car re-train)
    #   eval_idx + 5 : FINAL_SUMMARY
    # Ordre d'insertion = inverse pour préserver les indices

    insertions = [
        ("EDM_FINAL_SUMMARY", FINAL_SUMMARY, "edm_final_summary"),
        ("EDM_ABLATION_A_DAG_GATE", ABLATION_A_GATE, "edm_ablation_a"),
        ("EDM_ABLATION_C_NO_CROSSATTN", ABLATION_C_NOTE, "edm_ablation_c"),
        ("EDM_ABLATION_B_CFG", ABLATION_B_CFG, "edm_ablation_b"),
        ("EDM_RAPSD_PLOTS", RAPSD_PLOTS, "edm_rapsd_plots"),
    ]
    target_idx = eval_idx + 1

    for sentinel, content, cid in insertions:
        if _find_cell(cells, lambda s, sn=sentinel: sn in s) is None:
            cells.insert(target_idx, _make_code_cell(content, cid))
            n_changed += 1
            print(f"  + validation {sentinel} inséré en cell {target_idx}")
        else:
            print(f"  = validation {sentinel} déjà présent")

    if n_changed > 0:
        VAL_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return n_changed


def main() -> int:
    print("=== Training notebook ===")
    n1 = patch_training_notebook()
    print("\n=== Validation notebook ===")
    n2 = patch_validation_notebook()
    print(f"\n{n1 + n2} modification(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
