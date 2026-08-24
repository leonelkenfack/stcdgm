# >>> Cell 8 : gel de l'etage 1 + cache pour l'etage 2
from st_cdgm.training.two_stage import freeze_stage1, precompute_stage1_outputs

freeze_stage1(encoder, rcn_cell, regression_head, *([bg_head] if bg_head else []))
rcn_cell.A_dag.requires_grad_(False)
if rcn_cell.A_inst is not None:
    rcn_cell.A_inst.requires_grad_(False)
print("etage 1 gele, A_dag et A(0) compris - le DAG devient une feature OOD assumee")

# L'ancre mise en cache est mu = p*alpha*beta reexprimee en residu log1p,
# pas la projection 1 canal du decodeur (qui n'est plus entrainee sous A3).
cache = precompute_stage1_outputs(
    encoder=encoder, rcn_runner=rcn_runner, regression_head=regression_head,
    train_dataset=train_dataset,
    iterate_batches_fn=lambda s: convert_sample_v8(s, builder, DEVICE),
    device=DEVICE, bg_head=bg_head)
print({k: tuple(v.shape) for k, v in cache.items()})

# sigma_data APRES le cache, et mesure sur la VRAIE cible de la diffusion.
# calibrate_sigma_data_variant n'accepte pas bg_head : elle passerait par
# regression_head(H_T), donc par upsample[-1] - la projection 1 canal que la
# tete BG court-circuite et qui n'a JAMAIS recu de gradient. sigma_data aurait
# ete l'ecart-type d'une projection aleatoire, et il alimente c_skip/c_out/c_in
# du preconditionneur EDM.
_delta = cache["delta_target"]
_mask = cache["valid_mask"].bool()
SIGMA_DATA = float(_delta[_mask].std())
print(f"sigma_data = {SIGMA_DATA:.5f}  (ecart-type du residu reellement diffuse)")

# --- CALIBRATION DE mu_HR : le controle qui doit tomber AVANT l'etage 2 -----
# delta_target = t - mu, ou t = HR_log - baseline_log est le residu VRAI. Si mu
# sur-estime l'amplitude de t, la diffusion passe sa capacite a ANNULER mu au
# lieu d'ajouter du detail — c'est le risque n°1 pre-enregistre (V6' prereg,
# `risk_1_and_contingency`), avec pour declencheur corr(D_y, mu) < -0,3.
# Ce diagnostic-ci est cote DONNEES : il se lit sur le cache, sans entrainer,
# donc avant de payer les heures de l'etage 2.
_pas = max(1, cache["mu_HR"].shape[0] // 700)      # ~700 fenetres suffisent
_m = cache["valid_mask"][::_pas].bool()
_mu_c = cache["mu_HR"][::_pas][_m].double()
_dl_c = cache["delta_target"][::_pas][_m].double()
_t_c = _dl_c + _mu_c
_sig_t, _sig_mu = float(_t_c.std()), float(_mu_c.std())
_corr = lambda a, b: float(((a - a.mean()) @ (b - b.mean()))
                           / ((a - a.mean()).norm() * (b - b.mean()).norm()))
# a_opt : le facteur d'echelle qui MINIMISE ||t - a*mu||. Loin de 1 = mu mal
# calibre en amplitude, ce qui est un defaut different d'un mu peu informatif.
_a_opt = float((_t_c @ _mu_c) / (_mu_c @ _mu_c))
# R2 de mu comme predicteur de t. NEGATIF = mu fait pire que ne rien predire :
# l'etage 2 devrait alors commencer par defaire l'etage 1.
_r2 = 1.0 - float((_t_c - _mu_c).pow(2).sum() / _t_c.pow(2).sum())
print(f"calibration mu_HR : sigma(t)={_sig_t:.4f} sigma(mu)={_sig_mu:.4f} "
      f"(rapport {_sig_mu / max(_sig_t, 1e-9):.2f})")
print(f"                    corr(t,mu)={_corr(_t_c, _mu_c):+.3f} | "
      f"a_opt={_a_opt:.3f} | R2={_r2:+.3f}")
print(f"                    corr(delta,mu)={_corr(_dl_c, _mu_c):+.3f} "
      f"(seuil pre-enregistre : -0,30)")
if _r2 < 0.0 and not os.environ.get("V8_IGNORE_MU_CALIB"):
    raise RuntimeError(
        f"R2(mu_HR) = {_r2:+.3f} < 0 : l'etage 1 predit PIRE que zero, la "
        f"diffusion devrait d'abord defaire mu (a_opt={_a_opt:.3f}, "
        f"sigma(mu)/sigma(t)={_sig_mu / max(_sig_t, 1e-9):.2f}). Entrainer "
        f"l'etage 2 par-dessus coute des heures pour un resultat ininterpretable. "
        f"Reprendre l'etage 1, ou basculer sur la variante pre-enregistree "
        f"V6'.1 (delta = HR - baseline, mu en conditionnement seul). "
        f"V8_IGNORE_MU_CALIB=1 pour passer outre en connaissance de cause.")
if _corr(_dl_c, _mu_c) < -0.3:
    print("ATTENTION : declencheur pre-enregistre V6'.1 ATTEINT sur les donnees.")
del _mu_c, _dl_c, _t_c, _m

torch.save({"sigma_data": SIGMA_DATA, "node_types": NODE_TYPES,
            "v8": OmegaConf.to_container(V8)},
           CKPT_DIR / "stage1_frozen_meta.pth")
