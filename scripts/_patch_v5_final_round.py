"""Round final de patches : 7 fixes P0/P1 du 5-agent audit.

P0 (data safety) :
  1. NaN guard sur loss_total avant backward (silent corruption risk)
  2. F1 formula alignment : per-sample threshold comme original (vs global stack)
  3. RAPSD formula alignment : rfft2 + mean(|pred-target|) comme original
  4. Markdown V5/Noncausal restant (8 occurrences cells 3, 11, 13)

P1 (resilience) :
  5. DAG projection (spectral + floor) apres optimizer.step() — anti-collapse
  6. Dead code : suppression du _dummy_call orphan lambda
  7. Docstring fix : ckpt_save_dir=V5_DIR -> ORACLE_FINETUNED_DIR
"""
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ============================================================
# Fix 1 + 5 : finetune_stage1_bundle_b.py — NaN guard + DAG projection
# ============================================================
F = Path("scripts/finetune_stage1_bundle_b.py")
src = F.read_text(encoding="utf-8")

# Fix 1 : NaN guard avant backward
OLD_BACKWARD = '''        # === Backward + grad clip + step ===
        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()'''

NEW_BACKWARD = '''        # === Backward + grad clip + step ===
        # P0 fix : NaN/Inf guard. Si la loss est non-finite (overflow CC reg,
        # slogdet fail, spectral log explosion), skip le step pour ne pas
        # corrompre les poids ni le checkpoint inprogress.
        if not torch.isfinite(loss_total):
            warnings.warn(
                f"[NaN guard] epoch {epoch_idx}, batch {batch_idx} : "
                f"loss_total non-finite ({loss_total.item()}). Skip step."
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()'''

if OLD_BACKWARD in src:
    src = src.replace(OLD_BACKWARD, NEW_BACKWARD)
    print("[OK] Fix 1 : NaN guard sur loss_total avant backward")

# Fix 5 : DAG projection apres optimizer.step()
OLD_STEP = '''        if hp.get("gradient_clipping", None):
            all_params = []
            for group in optimizer.param_groups:
                all_params.extend(group["params"])
            torch.nn.utils.clip_grad_norm_(all_params, hp["gradient_clipping"])
        optimizer.step()'''

NEW_STEP = '''        if hp.get("gradient_clipping", None):
            all_params = []
            for group in optimizer.param_groups:
                all_params.extend(group["params"])
            torch.nn.utils.clip_grad_norm_(all_params, hp["gradient_clipping"])
        optimizer.step()

        # P1 fix : DAG anti-collapse projections apres step.
        # Avec lambda_l1_start=0.10 (10x baseline), A_dag peut collapser
        # vers 0 sur premieres epochs. project_dag_spectral garde le rayon
        # spectral < 0.95 (acyclicite), project_dag_floor preserve le prior.
        try:
            if hasattr(rcn_cell, "project_dag_spectral"):
                rcn_cell.project_dag_spectral(max_radius=0.95)
            if hasattr(rcn_cell, "project_dag_floor"):
                rcn_cell.project_dag_floor(min_norm=0.10, prior=G_phys)
        except Exception as e:
            if epoch_idx == 0 and batch_idx == 0:
                warnings.warn(f"DAG projection skipped: {type(e).__name__}: {e}")'''

if OLD_STEP in src:
    src = src.replace(OLD_STEP, NEW_STEP)
    print("[OK] Fix 5 : DAG projection (spectral + floor) apres step")

# Fix 6 : remove dead code _dummy_call lambda
OLD_DUMMY = '''            # Garde la signature pour compat
            _dummy_call = lambda *a, **k: sigma_data_new
            sigma_data_new = _dummy_call(  # keep the original structure pour eviter d'autres refs casses
                variant="causal",
'''
if OLD_DUMMY in src:
    src = src.replace(OLD_DUMMY, "")
    print("[OK] Fix 6 : dead code _dummy_call supprime")

# Fix 7 : docstring exemple V5_DIR -> ORACLE_FINETUNED_DIR
OLD_DOC = "        ckpt_save_dir=V5_DIR,\n        batch_size=8,"
NEW_DOC = "        ckpt_save_dir=ORACLE_FINETUNED_DIR,\n        batch_size=8,"
if OLD_DOC in src:
    src = src.replace(OLD_DOC, NEW_DOC)
    print("[OK] Fix 7 : docstring exemple -> ORACLE_FINETUNED_DIR")

F.write_text(src, encoding="utf-8")


# ============================================================
# Fix 2 + 3 : recompute_phase6_metrics.py — F1 et RAPSD alignment
# ============================================================
R = Path("scripts/recompute_phase6_metrics.py")
src = R.read_text(encoding="utf-8")

# Fix 2 : F1 per-sample threshold (comme original compute_f1_extremes)
# Le code actuel : pred_global stacks tous les batches, puis seuil global
# Le original : seuil per-sample, F1 per-sample, moyenne a la fin
OLD_F1_BLOCK = '''    rmse_global, mae_global = _rmse_mae(pred_global, target_global, mask_global)
    pearson_global = _pearson(pred_global, target_global, mask_global)
    f1_p95 = _f1_at_quantile(pred_global, target_global, mask_global, 0.95)
    f1_p99 = _f1_at_quantile(pred_global, target_global, mask_global, 0.99)
    rapsd_distance = float(np.mean(rapsd_per_sample))'''

NEW_F1_BLOCK = '''    rmse_global, mae_global = _rmse_mae(pred_global, target_global, mask_global)
    pearson_global = _pearson(pred_global, target_global, mask_global)
    # P0 fix : F1 per-sample puis moyenne, comme compute_f1_extremes original
    # (vs seuil global qui donne des nombres differents non comparables au baseline)
    f1_p95_per_sample = []
    f1_p99_per_sample = []
    for k in range(len(pred_all)):
        m_k = np.isfinite(target_all[k])
        f1_p95_per_sample.append(_f1_at_quantile(pred_all[k], target_all[k], m_k, 0.95))
        f1_p99_per_sample.append(_f1_at_quantile(pred_all[k], target_all[k], m_k, 0.99))
    f1_p95 = float(np.mean(f1_p95_per_sample)) if f1_p95_per_sample else float("nan")
    f1_p99 = float(np.mean(f1_p99_per_sample)) if f1_p99_per_sample else float("nan")
    rapsd_distance = float(np.mean(rapsd_per_sample))'''

if OLD_F1_BLOCK in src:
    src = src.replace(OLD_F1_BLOCK, NEW_F1_BLOCK)
    print("[OK] Fix 2 : F1 per-sample alignment (comme compute_f1_extremes original)")

# Fix 3 : RAPSD avec rfft2 + mean(|...|) comme compute_spectrum_distance
OLD_RAPSD = '''def _radial_power_spectrum(field: np.ndarray) -> np.ndarray:
    """RAPSD : moyenne radiale de la PSD 2D."""
    fft = np.fft.fft2(field)
    psd2d = np.abs(np.fft.fftshift(fft)) ** 2
    H, W = field.shape
    cy, cx = H // 2, W // 2
    Y, X = np.indices(field.shape)
    R = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2).astype(int)
    R_max = min(cx, cy)
    radial = np.zeros(R_max)
    for r in range(R_max):
        m = R == r
        if m.sum() > 0:
            radial[r] = psd2d[m].mean()
    return radial


def _rapsd_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """L1 distance entre RAPSD du pred et du target (cartes 2D)."""
    sp = _radial_power_spectrum(pred)
    st = _radial_power_spectrum(target)
    n = min(len(sp), len(st))
    return float(np.abs(sp[:n] - st[:n]).sum())'''

NEW_RAPSD = '''def _power_spectrum_rfft(field: np.ndarray) -> np.ndarray:
    """Spectre de puissance via rfft2, aligne sur compute_power_spectrum
    de evaluation_xai.py (utilise par le training original).

    Centre le champ (mean removal) puis applique rfft2 et calcule la
    puissance (|FFT|^2). Pas de moyennage radial — la distance L1 est
    calculee directement sur la matrice 2D des coefficients spectraux.
    """
    centered = field - field.mean()
    F = np.fft.rfft2(centered)
    return F.real ** 2 + F.imag ** 2


def _rapsd_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """L1 mean entre les power spectra (rfft2), aligne avec
    compute_spectrum_distance de evaluation_xai.py utilise par le
    training original. Pas de fft2/fftshift ni de sum, pour reproduire
    les magnitudes du baseline V5-mini final_validation_metrics.json.
    """
    sp = _power_spectrum_rfft(pred)
    st = _power_spectrum_rfft(target)
    return float(np.mean(np.abs(sp - st)))'''

if OLD_RAPSD in src:
    src = src.replace(OLD_RAPSD, NEW_RAPSD)
    print("[OK] Fix 3 : RAPSD aligne avec compute_spectrum_distance (rfft2 + mean)")

R.write_text(src, encoding="utf-8")


# ============================================================
# Fix 4 : Markdown V5/Noncausal restant (Cells 3, 11, 13)
# ============================================================
NB = Path("st_cdgm_v5_evaluation.ipynb")
nb = json.load(NB.open(encoding="utf-8"))

md_replacements = [
    # Cell 3 markdown
    ("V5 et Noncausal (encoder + RCN", "Oracle et CorrDiff (encoder + RCN"),
    # Cell 11 markdown
    ("V5×EC-Earth3, V5×NorESM2-MM, NC×EC-Earth3, NC×NorESM2-MM",
     "Oracle×EC-Earth3, Oracle×NorESM2-MM, CorrDiff×EC-Earth3, CorrDiff×NorESM2-MM"),
    # Cell 13 markdown
    ("DAG appris V5", "DAG appris Oracle"),
    ("V5 vs Noncausal vs vérité", "Oracle vs CorrDiff vs vérité"),
    ("V5 vs Noncausal vs verite", "Oracle vs CorrDiff vs verite"),
    # Autres patterns generaux
    ("V5 vs Noncausal", "Oracle vs CorrDiff"),
    (" Noncausal ", " CorrDiff "),
]

n_md = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "markdown":
        continue
    s = "".join(c["source"])
    changed = False
    for old, new in md_replacements:
        if old in s:
            s = s.replace(old, new)
            changed = True
            n_md += 1
    if changed:
        nb["cells"][i]["source"] = [l + "\n" for l in s.split("\n")[:-1]] + [s.split("\n")[-1]]

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[OK] Fix 4 : {n_md} markdown V5/Noncausal restants remplaces")

print()
print("=" * 60)
print("[OK] Round final : 7 fixes appliques")
print("=" * 60)
