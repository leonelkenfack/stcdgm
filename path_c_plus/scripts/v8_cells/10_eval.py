# >>> Cell 10 : evaluation IN-PROTOCOL (celle du 3-way V6' / ORACLE / CorrDiff)
# Le but de cette cellule n'est PAS de produire "des metriques" mais de produire
# les MEMES metriques, dans les MEMES conditions, que celles deja mesurees sur
# les autres modeles — sinon la Cell 11 comparerait des protocoles, pas des
# modeles. Tout ce qui suit est donc contraint :
#   evaluate_ensemble        la meme fonction (composition mm PAR MEMBRE)
#   K=32, 24 pas, cfg 0.0    les reglages du 3-way
#   graines 1000+k           les memes tirages
#   split de test complet     au meme stride
#   clim per-pixel partagee  le meme .npz quand il est disponible
import time as _time
from st_cdgm.evaluation.eval_metrics_dual_convention import (
    evaluate_ensemble, to_mm_day)
from st_cdgm.evaluation.two_stage_inference import sample_once_edm

diffusion.eval()
METRICS_PATH = RESULTS_DIR / "v8_metrics_inprotocol.json"

# --- 1. etage 1 sur TOUT le split de test ---------------------------------
# Meme chemin que le cache d'entrainement (Cell 8), tete BG comprise : mu doit
# designer la meme quantite des deux cotes, sinon delta n'a pas le meme sens.
test_cache = precompute_stage1_outputs(
    encoder=encoder, rcn_runner=rcn_runner, regression_head=regression_head,
    train_dataset=test_dataset,
    iterate_batches_fn=lambda s: convert_sample_v8(s, builder, DEVICE),
    device=DEVICE, bg_head=bg_head)
mu_all, base_all = test_cache["mu_HR"], test_cache["baseline_log"]
delta_all = test_cache["delta_target"]
N_TEST = int(mu_all.shape[0])
print(f"test : {N_TEST} fenetres (split complet, stride {_STRIDE})")

# --- 2. climatologie per-pixel (Convention A, ETCCDI) ---------------------
# Reutiliser le .npz des runs precedents quand il existe : un seuil p99 estime
# sur un echantillon different donne un F1 different a modele EGAL. Les clefs
# `clim_p95`/`clim_p99` sont celles ecrites par les runs 9-node et V6'.
_cands = [Path("results/clim_p95_p99.npz")]
if IN_COLAB:
    _cands = [Path(DRIVE_ROOT) / "oracle_9node/seed_42/phase8/clim_p95_p99.npz",
              Path(DRIVE_ROOT) / "ckpt_v2_corrdiff_normal/clim_p95_p99.npz",
              Path(DRIVE_ROOT) / "oracle_v6_prime/seed_42/clim_p95_p99.npz"] + _cands
_cp = next((p for p in _cands if p.exists()), None)
if _cp is not None:
    _z = np.load(_cp)
    clim95 = torch.from_numpy(_z["clim_p95"].astype("float32"))
    clim99 = torch.from_numpy(_z["clim_p99"].astype("float32"))
    CLIM_SOURCE = str(_cp)
else:
    # Recompose le HR VRAI depuis le cache d'ENTRAINEMENT : baseline + mu +
    # delta_target redonne exactement la cible, independamment du modele. C'est
    # la periode d'entrainement qui sert de reference (standard ETCCDI) — la
    # calculer sur la verite de test definirait l'evenement extreme a partir des
    # echantillons servant a compter les succes.
    _hr_mm = to_mm_day(cache["baseline_log"] + cache["mu_HR"]
                       + cache["delta_target"]).squeeze(1).numpy()
    _p95 = np.nanpercentile(_hr_mm, 95.0, axis=0).astype("float32")
    _p99 = np.nanpercentile(_hr_mm, 99.0, axis=0).astype("float32")
    np.savez(RESULTS_DIR / "clim_p95_p99.npz", clim_p95=_p95, clim_p99=_p99)
    clim95, clim99 = torch.from_numpy(_p95), torch.from_numpy(_p99)
    CLIM_SOURCE = f"calculee sur {_hr_mm.shape[0]} jours d'entrainement"
    del _hr_mm
print(f"climatologie : {CLIM_SOURCE}")
print(f"  p95 med={float(clim95.median()):.2f} | p99 med={float(clim99.median()):.2f} mm/j")

# --- 3. echantillonnage par lots ------------------------------------------
# Un membre a la fois sur TOUT le split, par paquets de EVAL_BATCH. Echantillonner
# sample par sample (batch 1) multiplierait le nombre de forwards UNet par
# EVAL_BATCH : plusieurs heures au lieu de dizaines de minutes.
_EPOCHS_DONE = len(hist2)


def sample_ensemble(K):
    members, t0 = [], _time.time()
    with torch.no_grad():
        for k in range(K):
            torch.manual_seed(1000 + k)   # memes graines que le 3-way
            chunks = []
            for i0 in range(0, N_TEST, EVAL_BATCH):
                sl = slice(i0, min(i0 + EVAL_BATCH, N_TEST))
                chunks.append(sample_once_edm(
                    diffusion, mu_HR=mu_all[sl].to(DEVICE),
                    baseline_log=base_all[sl].to(DEVICE),
                    scheduler_type="edm_karras", num_steps=NUM_STEPS,
                    cfg_scale=CFG_SCALE).cpu())
            members.append(torch.cat(chunks, 0))
            if k == 0 or (k + 1) % 4 == 0:
                _el = _time.time() - t0
                print(f"  membre {k + 1}/{K} | {_el:.0f}s ecoule | "
                      f"ETA {_el / (k + 1) * (K - k - 1):.0f}s", flush=True)
    return torch.stack(members, 0)          # [K, N, 1, H, W] sur CPU


# PERSISTANCE. L'echantillonnage coute des dizaines de minutes ; relancer la
# cellule pour lire la table de la Cell 11 ne doit pas le refaire. On invalide
# quand meme le cache si l'etage 2 a avance depuis : sinon la table afficherait
# en silence les metriques d'un modele moins entraine.
_prev = json.load(open(METRICS_PATH)) if METRICS_PATH.exists() else None
if _prev and _prev.get("protocol", {}).get("stage2_epochs") == _EPOCHS_DONE \
        and _prev.get("protocol", {}).get("K") == K_VERDICT:
    res = _prev["metrics"]
    print(f"metriques relues ({METRICS_PATH}) - supprimer le fichier pour recalculer")
else:
    if _prev:
        print(f"cache de metriques perime (etage 2 : "
              f"{_prev.get('protocol', {}).get('stage2_epochs')} -> {_EPOCHS_DONE} "
              f"epoques) - recalcul")
    ens = sample_ensemble(K_VERDICT)
    try:
        res = evaluate_ensemble(ens.to(DEVICE), mu_all.to(DEVICE),
                                base_all.to(DEVICE), delta_all.to(DEVICE),
                                clim99.to(DEVICE), clim95.to(DEVICE))
    except torch.cuda.OutOfMemoryError:
        # L'ensemble fait ~0,7 Go et evaluate_ensemble en materialise deux
        # copies. Plutot que de perdre l'echantillonnage sur un OOM a la
        # derniere ligne, on refait le calcul sur CPU : plus lent, identique.
        torch.cuda.empty_cache()
        print("OOM GPU sur les metriques -> recalcul sur CPU")
        res = evaluate_ensemble(ens, mu_all, base_all, delta_all,
                                clim99, clim95)
    del ens
    json.dump({"metrics": res,
               "protocol": {"K": K_VERDICT, "num_steps": NUM_STEPS,
                            "cfg_scale": CFG_SCALE, "n_test": N_TEST,
                            "stride": int(_STRIDE), "gcm": "ACCESS-CM2",
                            "clim_source": CLIM_SOURCE,
                            "stage2_epochs": _EPOCHS_DONE,
                            "composition": "mm par membre (evaluate_ensemble)"},
               "v8_nominal": OmegaConf.to_container(V8)},
              open(METRICS_PATH, "w"), indent=2, default=float)
    print(f"ecrit : {METRICS_PATH}")

print()
for _k in ("conv_A_F1p99", "conv_B_F1p99", "rmse", "pearson_global", "crps"):
    print(f"  {_k:16s} = {res[_k]:.4f}")
