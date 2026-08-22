# >>> Cell 10 : evaluation + verdict pre-enregistre
from st_cdgm.evaluation.eval_metrics_dual_convention import (
    to_mm_day, compute_f1_both_conventions)
from st_cdgm.evaluation.two_stage_inference import sample_once_edm

diffusion.eval()
preds, truths = [], []
with torch.no_grad():
    for i, s in enumerate(test_dataset):
        if i >= N_EVAL:
            break
        b = convert_sample_to_batch(s, builder, DEVICE)
        t = b["residual"][-1].to(DEVICE)
        if t.dim() == 3:
            t = t.unsqueeze(0)
        bl = b["baseline"][-1].to(DEVICE)
        if bl.dim() == 3:
            bl = bl.unsqueeze(0)
        mu = predict_mu_hr(b, variant="causal", encoder=encoder, rcn_runner=rcn_runner,
                           regression_head=regression_head, builder=builder,
                           device=DEVICE, target_shape=t.shape[-2:], bg_head=bg_head)
        members = [sample_once_edm(diffusion, mu, bl, device=DEVICE)
                   for _ in range(K_ENSEMBLE)]
        # expm1 PAR MEMBRE puis moyenne. L'ordre inverse rouvrirait un ecart de
        # Jensen que A1 ne corrige pas : A1 porte sur la moyenne conditionnelle,
        # pas sur des tirages. Mesure sur ces donnees : -2 a -15 % au p99.
        preds.append(torch.stack([to_mm_day(bl + mu + d) for d in members]).mean(0).cpu())
        truths.append(to_mm_day(bl + t).cpu())
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{N_EVAL}")

pred = torch.cat(preds).squeeze()
truth = torch.cat(truths).squeeze()
clim99 = torch.quantile(truth.float(), 0.99, dim=0)
clim95 = torch.quantile(truth.float(), 0.95, dim=0)
f1 = compute_f1_both_conventions(pred, truth, clim99, clim95)
rmse = float(((pred - truth) ** 2).mean().sqrt())
print(f"\nF1@p99 : {f1}")
print(f"RMSE   : {rmse:.4f} mm/j")

# References mesurees dans les runs precedents, in-protocol.
REF = {"v5_per_gridpoint": 0.841, "noncausal_per_gridpoint": 0.816,
       "v5_pooled": 0.512, "corrdiff_pooled": 0.550}
verdict = {
    "f1": f1, "rmse_mm": rmse, "references": REF,
    "ensemble_K": K_ENSEMBLE, "n_eval": N_EVAL,
    "v8": OmegaConf.to_container(V8),
    "lecture": (
        "Cible V8 = PARITE in-distribution + gain OOD, pas un gain ID. Une "
        "regression ID de quelques pour cent est PREVUE et acceptee : le DAG "
        "gele est une feature OOD assumee (critique froide 2026-07-05). Le "
        "verdict se joue sur EC-Earth3, puis UNE SEULE FOIS sur le holdout "
        "NorESM2-MM. Un gain ID ici serait une bonne surprise, pas le critere."),
}
json.dump(verdict, open(RESULTS_DIR / "v8_verdict.json", "w"), indent=2, default=float)
print("\n=== ECRIT results/v8_verdict.json ===")
print(verdict["lecture"])
