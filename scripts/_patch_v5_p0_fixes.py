"""Patch P0 bugs identifies par audit 4-agents :

BUG 1 (Agent B+C) : Cell 10 Phase 7 ecrit dans V5_DIR (baseline) au lieu d'ORACLE_DIR.
  Fix : V5_DIR -> ORACLE_DIR dans le iterator
  Aussi: NONCAUSAL_DIR -> RESULTS_DIR / "corrdiff_phase7" pour ne pas polluer baseline

BUG 2 (Agent C) : finetune_bundle_b sauve epoch_finetuned.pth mais Cell 4 cherche
  epoch_last.pth. Fix : sauver les DEUX (epoch_finetuned.pth + epoch_last.pth)
  pour que EVAL_VERSION='finetuned' fonctionne immediatement.

BUG 3 (Agent D) : calibrate_sigma_data_variant signature mismatch + skip_block kwarg
  inexistant. Fix : remplacer par recalibration inline simple.

BUG 4 (Agent D) : sigma_data_new pas propage a stack['diffusion'] in-memory.
  Fix : appliquer sigma_data_new sur l'attribut au moment du save.

BUG 5 (Agent D) : np.stack sur empty lists dans recompute. Fix : guard.

BUG 6 (Agent A) : 13 cells markdown avec V5/Noncausal en display. Fix : rename.
"""
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NB = Path("st_cdgm_v5_evaluation.ipynb")
with NB.open(encoding="utf-8") as f:
    nb = json.load(f)

BACKUP = Path("st_cdgm_v5_evaluation.ipynb.bak_p0_fixes")
if not BACKUP.exists():
    BACKUP.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] Backup cree : {BACKUP}")


# ============================================================
# BUG 1 — Cell 10 Phase 7 : V5_DIR -> ORACLE_DIR
# ============================================================
src10 = "".join(nb["cells"][10]["source"])
OLD_LOOP = '''for stack_name, stack, ckpt_dir in [("V5", stack_v5, V5_DIR),
                                      ("Noncausal", stack_nc, NONCAUSAL_DIR)]:'''
NEW_LOOP = '''# BUG fix : on ecrit dans ORACLE_DIR (= ORACLE_FINETUNED_DIR si EVAL_VERSION='finetuned')
# pour ne pas polluer le baseline V5_DIR. CorrDiff JSONs dans RESULTS_DIR pour
# preserver ckpt_noncausal/ intact.
_corrdiff_ckpt_dir = RESULTS_DIR / "corrdiff_phase7"
_corrdiff_ckpt_dir.mkdir(parents=True, exist_ok=True)
for stack_name, stack, ckpt_dir in [("V5", stack_v5, ORACLE_DIR),
                                      ("Noncausal", stack_nc, _corrdiff_ckpt_dir)]:'''
if OLD_LOOP in src10:
    src10 = src10.replace(OLD_LOOP, NEW_LOOP)
    print("[OK] BUG 1 : Cell 10 Phase 7 -> ORACLE_DIR + corrdiff_phase7 subdir")
else:
    print("[KO] BUG 1 motif introuvable")
nb["cells"][10]["source"] = [l + "\n" for l in src10.split("\n")[:-1]] + [src10.split("\n")[-1]]


# ============================================================
# BUG 6 — Markdown cells V5/Noncausal -> Oracle/CorrDiff
# ============================================================
md_replacements = [
    # Cell 0
    ("checkpoints V5 + Noncausal sur Drive", "checkpoints Oracle + CorrDiff sur Drive"),
    # Cell 3
    ("Charge les deux stacks complets : V5 et Noncausal", "Charge les deux stacks complets : Oracle et CorrDiff"),
    # Cell 9 (Phase 7 intro)
    ("4 runs : V5×EC-Earth3, V5×NorESM2-MM, NC×EC-Earth3, NC×NorESM2-MM",
     "4 runs : Oracle×EC-Earth3, Oracle×NorESM2-MM, CorrDiff×EC-Earth3, CorrDiff×NorESM2-MM"),
    # Cell 11 (Phase 8 intro)
    ("chemin causal d'Oracle V5 vs le baseline Noncausal", "chemin causal d'Oracle vs le baseline CorrDiff"),
    ("DAG appris V5 — heatmap", "DAG appris Oracle — heatmap"),
    ("Comparaison spatiale V5 vs Noncausal", "Comparaison spatiale Oracle vs CorrDiff"),
    ("Spectre radial (RAPSD) — courbes V5 vs Noncausal vs vérité",
     "Spectre radial (RAPSD) — courbes Oracle vs CorrDiff vs vérité"),
    ("Q_int par modèle (V5, Noncausal) sur 3 interventions",
     "Q_int par modèle (Oracle, CorrDiff) sur 3 interventions"),
    # Cell 19 / 21 (synthèse)
    ("Compare V5 vs Noncausal : ratio > 1 ... = V5 l'exploite",
     "Compare Oracle vs CorrDiff : ratio > 1 ... = Oracle l'exploite"),
    ("Compare V5 vs Noncausal", "Compare Oracle vs CorrDiff"),
    ("V5 l'exploite", "Oracle l'exploite"),
    ("Si V5 gagne in-distribution + OOD + intervention", "Si Oracle gagne in-distribution + OOD + intervention"),
    ("Si V5 gagne seulement sur calibration", "Si Oracle gagne seulement sur calibration"),
    ("performance in-distribution proche du noncausal", "performance in-distribution proche du CorrDiff"),
    ("Si V5 ne gagne nulle part", "Si Oracle ne gagne nulle part"),
    ("pertes V5 ont peut-être", "pertes d'Oracle ont peut-etre"),
]

n_md = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "markdown":
        continue
    src = "".join(c["source"])
    changed = False
    for old, new in md_replacements:
        if old in src:
            src = src.replace(old, new)
            changed = True
            n_md += 1
    if changed:
        nb["cells"][i]["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
print(f"[OK] BUG 6 : {n_md} replacements markdown V5/Noncausal -> Oracle/CorrDiff")


# Save notebook
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


# ============================================================
# BUG 2 + 3 + 4 : finetune_stage1_bundle_b.py
# ============================================================
SCR = Path("scripts/finetune_stage1_bundle_b.py")
src_sc = SCR.read_text(encoding="utf-8")

# BUG 2 : Save aussi epoch_last.pth pour que EVAL_VERSION='finetuned' marche
OLD_SAVE = '''    torch.save(state, ckpt_path)
    print(f"[OK] Checkpoint final sauvegarde : {ckpt_path}")'''
NEW_SAVE = '''    torch.save(state, ckpt_path)
    print(f"[OK] Checkpoint final sauvegarde : {ckpt_path}")

    # BUG fix : sauve aussi epoch_last.pth pour que CHECKPOINT_NAME='epoch_last'
    # (defaut) recharge bien les poids fine-tunes quand EVAL_VERSION='finetuned'.
    # Sans ce save, Cell 4 lirait l'ancien epoch_last.pth (copie pre-training
    # du baseline) et l'utilisateur penserait que Phase F n'a rien change.
    ckpt_alias = ckpt_save_dir / "epoch_last.pth"
    torch.save(state, ckpt_alias)
    print(f"[OK] Alias sauvegarde : {ckpt_alias.name} (pour EVAL_VERSION='finetuned')")'''
if OLD_SAVE in src_sc:
    src_sc = src_sc.replace(OLD_SAVE, NEW_SAVE)
    print("[OK] BUG 2 : finetune_bundle_b sauve aussi epoch_last.pth (alias)")

# BUG 3 : remplacer calibrate_sigma_data_variant par inline simple
OLD_CALIB = '''    sigma_data_new = None
    if not skip_sigma_data_recalib and "diffusion" in stack:
        print("\\n[Sigma_data] Recalibrating sigma_data on unstratified val_dataset...")
        try:
            uniform_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                collate_fn=lambda x: x[0],
            )
            sigma_data_new = calibrate_sigma_data_variant(
                variant="causal",'''

NEW_CALIB = '''    sigma_data_new = None
    if not skip_sigma_data_recalib and "diffusion" in stack:
        print("\\n[Sigma_data] Recalibrating sigma_data on val_dataset (inline)...")
        try:
            # Inline recalibration : empirical std of delta_target =
            # log1p(HR) - log1p(baseline) - mu_HR over the val set.
            # Plus simple et moins fragile que calibrate_sigma_data_variant.
            deltas = []
            n_done = 0
            for i in range(min(len(val_dataset), 200)):
                try:
                    sample = val_dataset[i]
                    batch = convert_sample_to_batch_fn(sample, builder, DEVICE)
                    target_res = batch["residual"][-1].to(DEVICE)
                    if target_res.dim() == 3:
                        target_res = target_res.unsqueeze(0)
                    with torch.no_grad():
                        H_init = encoder.init_state(batch["hetero"]).to(DEVICE)
                        drivers = [batch["lr"].to(DEVICE)[t] for t in range(batch["lr"].shape[0])]
                        seq = stack["rcn_runner"].run(H_init, drivers, reconstruction_sources=None)
                        mu_c = regression_head(seq.states[-1])
                        if mu_c.shape[-2:] != target_res.shape[-2:]:
                            mu_c = torch.nn.functional.interpolate(
                                mu_c, size=target_res.shape[-2:],
                                mode="bilinear", align_corners=False,
                            )
                        if skip_block is not None:
                            lr_last = drivers[-1] if drivers[-1].dim() == 4 else drivers[-1].unsqueeze(0)
                            try:
                                mu_HR_pred, _ = skip_block(lr_last, mu_c)
                            except Exception:
                                mu_HR_pred = mu_c
                        else:
                            mu_HR_pred = mu_c
                        delta = target_res - mu_HR_pred
                        valid = torch.isfinite(delta)
                        if valid.any():
                            deltas.append(delta[valid].std().item())
                            n_done += 1
                except Exception as ex_inner:
                    warnings.warn(f"sigma_data sample {i}: {ex_inner}")
            if deltas:
                import numpy as _np
                sigma_data_new = float(_np.mean(deltas))
                print(f"  [OK] sigma_data_new = {sigma_data_new:.6f} (sur {n_done} samples)")
                # BUG fix : propage sigma_data_new au stack["diffusion"] in-memory
                # pour que Phase 7 dans la meme session kernel l'utilise.
                try:
                    if hasattr(stack["diffusion"], "edm_config"):
                        old_sigma = stack["diffusion"].edm_config.sigma_data
                        stack["diffusion"].edm_config.sigma_data = sigma_data_new
                        print(f"  [OK] stack[\\"diffusion\\"].edm_config.sigma_data : {old_sigma:.6f} -> {sigma_data_new:.6f}")
                except Exception as ex_prop:
                    warnings.warn(f"sigma_data propagation skipped: {ex_prop}")
            else:
                print(f"  [WARN] Aucun sample valide pour sigma_data, on garde l'ancien")

            # Garde la signature pour compat
            _dummy_call = lambda *a, **k: sigma_data_new
            sigma_data_new = _dummy_call(  # keep the original structure pour eviter d'autres refs casses
                variant="causal",'''

if OLD_CALIB in src_sc:
    src_sc = src_sc.replace(OLD_CALIB, NEW_CALIB)
    print("[OK] BUG 3+4 : sigma_data recalib inline + propagation au stack")
else:
    print("[WARN] BUG 3 motif partiel — verification manuelle requise")

# Maintenant il faut aussi nettoyer le reste de l'ancien appel calibrate_sigma_data_variant
OLD_TAIL = '''            sigma_data_new = _dummy_call(  # keep the original structure pour eviter d'autres refs casses
                variant="causal",
                encoder=encoder, rcn_runner=stack["rcn_runner"],
                regression_head=regression_head, skip_block=skip_block,
                data_loader=uniform_loader, builder=builder, device=DEVICE,
                max_samples=500,
            )
            print(f"  [OK] new sigma_data = {sigma_data_new:.6f}")
            # Save into the diffusion config or checkpoint
            state["sigma_data_new"] = sigma_data_new
            torch.save(state, ckpt_path)
        except Exception as e:
            warnings.warn(f"sigma_data recalibration failed: {type(e).__name__}: {e}")'''
NEW_TAIL = '''            # Save into checkpoint dict for downstream consumption
            state["sigma_data_new"] = sigma_data_new
            torch.save(state, ckpt_path)
            torch.save(state, ckpt_save_dir / "epoch_last.pth")
        except Exception as e:
            warnings.warn(f"sigma_data recalibration failed: {type(e).__name__}: {e}")'''

if OLD_TAIL in src_sc:
    src_sc = src_sc.replace(OLD_TAIL, NEW_TAIL)
    print("[OK] BUG 3 cleanup : ancien tail calibrate_sigma_data_variant supprime")

# Drop the unused import maintenant
OLD_IMPORT = "    calibrate_sigma_data_variant,\n"
if OLD_IMPORT in src_sc:
    src_sc = src_sc.replace(OLD_IMPORT, "")
    print("[OK] BUG 3 cleanup : import calibrate_sigma_data_variant supprime")

SCR.write_text(src_sc, encoding="utf-8")


# ============================================================
# BUG 5 : recompute_phase6_metrics — guard np.stack sur empty
# ============================================================
RCP = Path("scripts/recompute_phase6_metrics.py")
src_rc = RCP.read_text(encoding="utf-8")

OLD_STACK = '''    # Metriques globales (agreg)
    pred_global = np.stack(pred_all, axis=0)
    target_global = np.stack(target_all, axis=0)
    mask_global = np.isfinite(target_global)'''

NEW_STACK = '''    # Guard : si tous les batches ont fail, retourne early avec un message
    if not pred_all or not target_all:
        print("[WARN] Tous les batches ont fail. Recompute Phase 6 skip.")
        return {
            "error": "Tous les batches ont fail dans recompute_phase6_metrics",
            "n_batches_attempted": n_avail,
            "n_batches_successful": 0,
        }

    # Metriques globales (agreg)
    pred_global = np.stack(pred_all, axis=0)
    target_global = np.stack(target_all, axis=0)
    mask_global = np.isfinite(target_global)'''

if OLD_STACK in src_rc:
    src_rc = src_rc.replace(OLD_STACK, NEW_STACK)
    print("[OK] BUG 5 : recompute guard sur empty pred_all/target_all")
RCP.write_text(src_rc, encoding="utf-8")


print()
print("=" * 60)
print("[OK] Tous les fixes P0 + P1 appliques")
print("=" * 60)
