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

# QUI est non fini. `JensenCorrector.fit` ne sait dire que "0 pixels finis" :
# le compte porte sur bl+mu ET tg-mu, donc n'importe lequel des trois tableaux
# peut etre en cause et le message ne le dit pas. Deux cycles de debogage ont
# ete perdus a chercher du cote de mu alors que rien ne l'y designait.
_diag = {"mu": mu_tr, "cible": tg_tr, "baseline": bl_tr}
for _n, _a in _diag.items():
    print(f"  {_n:9s} fini sur {100 * float(np.isfinite(_a).mean()):5.1f} % des pixels")
_com = np.isfinite(mu_tr) & np.isfinite(tg_tr) & np.isfinite(bl_tr)
if _com.sum() == 0:
    _coupables = [_n for _n, _a in _diag.items() if not np.isfinite(_a).any()]
    # mu non fini PARTOUT ne peut venir que de p, alpha ou beta, donc des POIDS
    # de la tete : la baseline, elle, est finie sur la terre. Le dire ici evite
    # de chercher la cause dans le calcul de l'ancre — ce qui a deja coute deux
    # cycles.
    if "mu" in _coupables:
        # OU la finitude se perd-elle ? mu non fini partout peut venir de la
        # tete comme de TOUT ce qui la precede — un etat H_T diverge donne des
        # features NaN, donc p/alpha/beta NaN, donc mu NaN, et la tete serait
        # innocente. On remonte la chaine sur un echantillon plutot que de
        # designer un coupable par raisonnement.
        print("\nOU la finitude se perd (un echantillon) :")
        for _mod, _nom in zip(STAGE1_MODULES,
                              ("encodeur", "RCN", "decodeur", "tete BG")):
            _bad = [n for n, _p in _mod.named_parameters()
                    if not torch.isfinite(_p).all()]
            print(f"  poids {_nom:9s} : "
                  + (f"NON FINIS -> {_bad[:4]}" if _bad else "finis"))
        with torch.no_grad():
            _s0 = next(iter(train_dataset))
            _b0 = convert_sample_v8(_s0, builder, DEVICE)
            _bl0 = _b0["baseline"][-1].to(DEVICE)
            if _bl0.dim() == 3:
                _bl0 = _bl0.unsqueeze(0)
            _lr0 = _b0["lr"].to(DEVICE)
            _et = [("entree LR", _lr0), ("H(0)", encoder.init_state(_b0["hetero"]))]
            _H0 = _et[-1][1]
            _HT = rcn_runner.run(_H0, [_lr0[k] for k in range(_lr0.shape[0])],
                                 reconstruction_sources=None).states[-1]
            _et.append(("H_T", _HT))
            _f0 = regression_head(_HT, return_features=True)
            _et.append(("features", _f0))
            if bg_head is not None:
                _p0, _a0, _b0p = bg_head(_f0, baseline_log=_bl0)
                _et += [("p", _p0), ("alpha", _a0), ("beta", _b0p)]
            for _nom, _t in _et:
                _pc = 100.0 * float(torch.isfinite(_t).float().mean())
                print(f"  {_nom:12s} fini sur {_pc:5.1f} %"
                      + ("   <- RUPTURE ICI" if _pc == 0.0 else ""))
    raise RuntimeError(
        f"aucun pixel exploitable. Non fini PARTOUT : "
        f"{', '.join(_coupables) if _coupables else 'aucun seul, mais leurs '
        'masques ne se recouvrent nulle part'}. Voir le trace ci-dessus.")
print(f"  intersection exploitable : {100 * float(_com.mean()):.1f} % "
      f"({int(_com.sum()):,} pixels)")

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
