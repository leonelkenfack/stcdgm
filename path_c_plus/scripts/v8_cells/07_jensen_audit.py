# >>> Cell 7 : A1 - calibration de Jensen + audit de conformite de l'etage 1
from st_cdgm.evaluation.jensen import JensenCorrector
from st_cdgm.training.stage1_paths import predict_mu_hr

for _m in STAGE1_MODULES:
    _m.eval()

@torch.no_grad()
def collect(ds, n_max=400):
    """mu / cible / baseline en espace log1p, [N, H, W]."""
    mus, tgs, bls = [], [], []
    for i, s in enumerate(ds):
        if i >= n_max:
            break
        b = convert_sample_v8(s, builder, DEVICE)
        t = b["residual"][-1].to(DEVICE)
        if t.dim() == 3:
            t = t.unsqueeze(0)
        bl = b["baseline"][-1].to(DEVICE)
        if bl.dim() == 3:
            bl = bl.unsqueeze(0)
        mu = predict_mu_hr(b, variant="causal", encoder=encoder, rcn_runner=rcn_runner,
                           regression_head=regression_head, builder=builder,
                           device=DEVICE, target_shape=t.shape[-2:], bg_head=bg_head)
        mus.append(mu.reshape(t.shape[-2:]).cpu())
        tgs.append(t.reshape(t.shape[-2:]).cpu())
        bls.append(bl.reshape(t.shape[-2:]).cpu())
    return (torch.stack(mus).numpy(), torch.stack(tgs).numpy(), torch.stack(bls).numpy())

mu_tr, tg_tr, bl_tr = collect(train_dataset)
print("echantillon de calibration :", mu_tr.shape)

# s^2 calibre sur le TRAIN uniquement. C'est un parametre de calibration :
# l'estimer sur le test ferait fuiter la cible dans la metrique.
jc = JensenCorrector.fit(mu_tr, tg_tr, bl_tr) if V8.jensen else None
if jc is not None:
    print(jc)
    torch.save(jc.state_dict(), CKPT_DIR / "jensen.pt")

# --- Audit de conformite. C1 juge le biais APRES correction de Jensen :
# sinon il mesurerait l'inegalite de Jensen et non le modele. Un etage 1
# EXACT en log1p affiche ~22 % de biais conditionnel sans la correction.
mu_te, tg_te, bl_te = collect(test_dataset, n_max=N_EVAL)
x_mm = np.expm1(np.clip(bl_te + tg_te, -20, 20))

def cond_bias(pred_mm, min_par_bin=500):
    """Biais relatif par decile de la PREDICTION, pas de la verite.

    C'est une courbe de fiabilite : le biais conditionne a l'intensite PREVUE.
    A ne pas lire comme un biais conditionne a l'intensite observee - les deux
    divergent sur un domaine a fort gradient orographique.
    """
    q = np.unique(np.quantile(pred_mm, np.linspace(0, 1, 11)))
    if q.size < 2:
        return []                    # champ constant : aucun bin exploitable
    bid = np.clip(np.digitize(pred_mm.ravel(), q[1:-1]), 0, len(q) - 2)
    xr_, mr_ = x_mm.ravel(), pred_mm.ravel()
    return [100 * (xr_[bid == k].mean() - mr_[bid == k].mean())
            / max(xr_[bid == k].mean(), 1e-9)
            for k in range(len(q) - 1) if (bid == k).sum() > min_par_bin]


naive_mm = np.expm1(np.clip(bl_te + mu_te, -20, 20))
corr_mm = jc.to_mm(mu_te, bl_te, delta=0.0).numpy() if jc is not None else naive_mm
_bn, _bc = cond_bias(naive_mm), cond_bias(corr_mm)
print()

if not _bn or not _bc:
    # Aucun bin exploitable = champ predit quasi CONSTANT. C'est un RESULTAT,
    # pas un plantage : soit l'echantillon est trop petit, soit l'etage 1
    # s'est effondre sur une prediction plate. Ce second cas est un mode de
    # defaillance documente de cette architecture - la cellule d'audit doit le
    # SIGNALER, pas lever une exception qui le masque.
    b_naive = b_corr = float("nan")
    _cause = "echantillon trop petit" if len(mu_te) < 50 else "ETAGE 1 PLAT"
    print(f"C1 NON CALCULABLE : moins de 2 deciles distincts de plus de 500 "
          f"pixels sur {len(mu_te)} echantillons. Ecart-type de la prediction "
          f"= {float(np.std(naive_mm)):.4f} mm/j  ->  {_cause}")
else:
    b_naive, b_corr = max(map(abs, _bn)), max(map(abs, _bc))
    print(f"biais conditionnel max : naif {b_naive:5.1f} %  ->  "
          f"corrige {b_corr:5.1f} %")
    print(f"C1 {'PASS' if b_corr < 5.0 else 'FAIL'} (seuil 5 %) | "
          f"part imputable a Jensen : {b_naive - b_corr:.1f} points")

json.dump({"C1_bias_corrected_pct": float(b_corr),
           "C1_bias_naive_pct": float(b_naive),
           "C1_jensen_share_pct": float(b_naive - b_corr),
           "C1_pass": (bool(b_corr < 5.0) if b_corr == b_corr else None),
           "C1_n_bins": len(_bc)},
          open(RESULTS_DIR / "v8_stage1_audit.json", "w"), indent=2)
