# >>> Cell 10 : evaluation + verdict pre-enregistre
from st_cdgm.evaluation.eval_metrics_dual_convention import (
    to_mm_day, compute_f1_both_conventions)
from st_cdgm.evaluation.two_stage_inference import sample_once_edm

diffusion.eval()

# Etaler les N_EVAL echantillons sur TOUTE la periode de test. Prendre les
# N_EVAL premiers jours consecutifs donnerait une tranche fortement
# autocorrelee (les episodes pluvieux durent plusieurs jours) et ignorerait la
# seconde annee de test.
_n_test = sum(1 for _ in test_dataset)
_stride = max(1, _n_test // max(N_EVAL, 1))
print(f"test : {_n_test} jours disponibles, {N_EVAL} echantillons pris tous les "
      f"{_stride} jours (couverture {100 * min(N_EVAL * _stride, _n_test) / _n_test:.0f} %)")

preds, truths = [], []
with torch.no_grad():
    for i, s in enumerate(test_dataset):
        if i % _stride:
            continue
        if len(preds) >= N_EVAL:
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
        # sample_once_edm ne prend pas de `device` : les tenseurs portent deja
        # le leur. num_steps vient du YAML, pas d'un defaut cache.
        members = [sample_once_edm(
            diffusion, mu_HR=mu, baseline_log=bl, scheduler_type="edm_karras",
            num_steps=int(CONFIG.diffusion.get("eval_num_steps",
                                               CONFIG.diffusion.steps)),
            cfg_scale=float(CONFIG.diffusion.get("cfg_scale", 1.0)))
            for _ in range(K_ENSEMBLE)]
        # expm1 PAR MEMBRE puis moyenne. L'ordre inverse rouvrirait un ecart de
        # Jensen que A1 ne corrige pas : A1 porte sur la moyenne conditionnelle,
        # pas sur des tirages. Mesure sur ces donnees : -2 a -15 % au p99.
        preds.append(torch.stack([to_mm_day(bl + mu + d) for d in members]).mean(0).cpu())
        truths.append(to_mm_day(bl + t).cpu())
        if len(preds) % 50 == 0:
            print(f"  {len(preds)}/{N_EVAL}")

pred = torch.cat(preds).squeeze()
truth = torch.cat(truths).squeeze()

# CLIMATOLOGIE DE LA PERIODE D'ENTRAINEMENT (convention ETCCDI, et celle avec
# laquelle les references 0,841 / 0,816 ont ete produites). La calculer sur la
# verite de TEST reviendrait a definir l'evenement extreme a partir des memes
# echantillons qu'on utilise pour compter les succes : le F1 en serait gonfle,
# et la comparaison aux references ne porterait plus sur la meme quantite.
import xarray as _xr

_CLIM = RESULTS_DIR / "clim_train_p95_p99.npz"
if _CLIM.exists():
    _z = np.load(_CLIM)
    clim95, clim99 = torch.from_numpy(_z["p95"]), torch.from_numpy(_z["p99"])
    print(f"climatologie relue : {_CLIM}")
else:
    _dh = _xr.open_dataset(guard(HR_PATH), decode_times=True)
    _tr = _dh["pr"].sel(time=slice(CONFIG.data.train_start_date,
                                   CONFIG.data.train_end_date))
    # 1 jour sur 2 : ~5500 jours suffisent pour un p99 par pixel (55 depassements)
    # et divisent par deux l'empreinte memoire sur Colab.
    _tr = _tr.isel(time=slice(None, None, 2)).values.astype("float32")
    print(f"climatologie calculee sur {_tr.shape[0]} jours d'entrainement")
    _p95 = np.nanquantile(_tr, 0.95, axis=0).astype("float32")
    _p99 = np.nanquantile(_tr, 0.99, axis=0).astype("float32")
    np.savez(_CLIM, p95=_p95, p99=_p99)
    clim95, clim99 = torch.from_numpy(_p95), torch.from_numpy(_p99)
    _dh.close()
    del _tr
print(f"seuils : p95 med={float(clim95.median()):.2f} mm/j | "
      f"p99 med={float(clim99.median()):.2f} mm/j")
f1 = compute_f1_both_conventions(pred, truth, clim99, clim95)
# RMSE masquee. Un echantillonnage de diffusion instable peut produire des
# valeurs non finies ; une RMSE faite main renverrait alors nan sans dire
# pourquoi. Observe sur un smoke ou l'etage 2 n'avait pas ete entraine.
_fin = torch.isfinite(pred) & torch.isfinite(truth)
_bad = int((~torch.isfinite(pred)).sum())
if _bad:
    print(f"ATTENTION : {_bad} pixels predits non finis "
          f"({100 * _bad / pred.numel():.2f} %) - echantillonnage instable, "
          f"les metriques ci-dessous portent sur les pixels valides seulement")
rmse = float(((pred[_fin] - truth[_fin]) ** 2).mean().sqrt())
print(f"\nF1@p99 : {f1}")
print(f"RMSE   : {rmse:.4f} mm/j")

# References mesurees dans les runs precedents, in-protocol.
REF = {"v5_per_gridpoint": 0.841, "noncausal_per_gridpoint": 0.816,
       "v5_pooled": 0.512, "corrdiff_pooled": 0.550}
verdict = {
    "f1": f1, "rmse_mm": rmse, "references": REF,
    "ensemble_K": K_ENSEMBLE, "n_eval": N_EVAL,
    "n_pixels_non_finis": _bad,
    "v8_nominal": OmegaConf.to_container(V8),
    # Configuration EFFECTIVE : certains interrupteurs en desactivent d'autres
    # (diagonal_driver et edge_prior exigent free_nodes). Enregistrer seulement
    # le dict nominal ferait croire qu'une brique etait active alors qu'elle
    # avait ete neutralisee en silence.
    "v8_effectif": {
        "free_nodes": bool(V8.free_nodes),
        "driver_routing": rcn_cell.driver_routing,
        "instantaneous": rcn_cell.A_inst is not None,
        "edge_prior": edge_prior is not None,
        "query_mode": regression_head.query_mode,
        "bernoulli_gamma": bg_head is not None,
        "jensen": jc is not None,
    },
    "protocole": {
        "clim_source": "periode d'entrainement (ETCCDI)",
        "n_test_disponibles": int(_n_test), "stride": int(_stride),
        "reference_K": 64, "reference_num_steps": 32,
    },
    "lecture": (
        "ATTRIBUTION : avec plusieurs interrupteurs actifs, ce run ne dit RIEN "
        "sur la contribution de chaque brique. Il faut les runs P1 a un seul "
        "changement pour cela. "
        "COMPARABILITE : les references ont ete produites a K=64 et 32 pas de "
        "diffusion ; a K plus faible la moyenne d'ensemble est plus bruitee. "
        "Un ecart de 1-2 points n'est pas interpretable sans intervalle de "
        "confiance par blocs ni plusieurs graines. "
        "Cible V8 = PARITE in-distribution + gain OOD, pas un gain ID. Une "
        "regression ID de quelques pour cent est PREVUE et acceptee : le DAG "
        "gele est une feature OOD assumee (critique froide 2026-07-05). Le "
        "verdict se joue sur EC-Earth3, puis UNE SEULE FOIS sur le holdout "
        "NorESM2-MM. Un gain ID ici serait une bonne surprise, pas le critere."),
}
json.dump(verdict, open(RESULTS_DIR / "v8_verdict.json", "w"), indent=2, default=float)
print("\n=== ECRIT results/v8_verdict.json ===")
print(verdict["lecture"])
