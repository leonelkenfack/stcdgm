# >>> BS44 — EVAL ALIGNEE cGAN (CDD / Rx1Day / R10 / saisonnier / PSD)
# Aligne le protocole d'evaluation sur celui du cGAN de Rampal (fonctions
# vendorisees dans st_cdgm.evaluation.cgan_aligned_metrics). Produit
# aligned_metrics_<GCM>_<run_variant>.json dans le dossier checkpoint, consomme
# par st_cdgm_causal_vs_noncausal_comparison.ipynb (section OOD).
#
# [!] Les indices cGAN (CDD, Rx1Day, R10, saisonnier) exigent une serie
#     temporelle CONTINUE par pixel. Definir EVAL_GCM_TAG + un dataloader
#     ORDONNE couvrant toute la periode du GCM :
#       - in-distribution : ACCESS-CM2 (val ordonne, sans shuffle)
#       - OOD            : EC-Earth3 / NorESM2-MM (data/raw/test/, pipeline
#                          pointe sur le fichier du GCM cible)
#     Reconstruction du champ HR complet (mm/jour) :
#       pr_mm = expm1(baseline_log + (mu_HR + delta_hat))
#     => on additionne baseline_log AVANT expm1 (espace log1p du projet).
from st_cdgm.evaluation.aligned_eval import run_aligned_eval
import numpy as _np44, torch as _t44

EVAL_GCM_TAG = str(globals().get("EVAL_GCM_TAG", "ACCESS-CM2"))
EVAL_IN_DIST = bool(globals().get("EVAL_IN_DIST", True))
_aligned_loader = globals().get("ALIGNED_EVAL_LOADER", globals().get("val_dataloader", None))
_run_variant44 = str(CONFIG.get("two_stage", {}).get("run_variant", "causal"))
_ck44 = Path(str(globals().get("_ckpt_dir", globals().get("CKPT_SAVE_DIR", "results"))))

if _aligned_loader is None:
    print("[BS44] Aucun dataloader (ALIGNED_EVAL_LOADER / val_dataloader). Skip.")
else:
    print(f"[BS44] Eval alignee cGAN | GCM={EVAL_GCM_TAG} | in_dist={EVAL_IN_DIST} "
          f"| variant={_run_variant44}")
    _Kal = int(globals().get("ALIGNED_K_SAMPLES", min(int(globals().get("K_SAMPLES", 16)), 16)))
    _pred_seq, _truth_seq, _time_seq = [], [], []
    _n_skip_time = 0
    with _t44.no_grad():
        for _conv in iterate_batches(_aligned_loader, builder, DEVICE):
            for _b in _conv:
                _cond, _muHR, _blog, _tgt = _build_inputs(_b)          # FINAL_VALIDATION helper
                _ens = _t44.stack([_sample_once(_cond, _muHR, _blog)
                                   for _ in range(_Kal)], dim=0).mean(dim=0)  # residu moyen
                # champ HR complet en log1p = baseline_log + (mu_HR + delta_hat)
                _full_log = (_blog if _blog is not None else 0.0) + \
                            (_muHR if _muHR is not None else 0.0) + _ens
                _truth_log = (_blog if _blog is not None else 0.0) + \
                             (_muHR if _muHR is not None else 0.0) + _tgt
                for _i in range(_full_log.shape[0]):
                    _pred_seq.append(_full_log[_i].squeeze().detach().float().cpu().numpy())
                    _truth_seq.append(_truth_log[_i].squeeze().detach().float().cpu().numpy())
                # temps : derniere date de la sequence du sample si dispo
                _tt = _b.get("time", None)
                if _tt is not None:
                    _arr = _np44.atleast_1d(_np44.asarray(_tt))
                    _time_seq.append(_arr.ravel()[-1])
                else:
                    _n_skip_time += 1

    _T = len(_pred_seq)
    if _T == 0:
        print("[BS44] Aucune prediction collectee. Skip.")
    else:
        # Axe temps : reel si dispo, sinon journalier synthetique (CDD/saison
        # restent definissables ; a confirmer sur Colab avec les vraies dates).
        if len(_time_seq) == _T:
            _times = _np44.asarray(_time_seq, dtype="datetime64[ns]")
        else:
            print(f"[BS44] [!] {_n_skip_time} samples sans 'time' -> axe synthetique journalier.")
            _times = _np44.arange(_T, dtype="datetime64[D]").astype("datetime64[ns]")
        _out44 = _ck44 / f"aligned_metrics_{EVAL_GCM_TAG}_{_run_variant44}.json"
        _res44 = run_aligned_eval(
            pred_fields=_pred_seq, truth_fields=_truth_seq, times=_times,
            out_path=_out44, gcm=EVAL_GCM_TAG, run_variant=_run_variant44,
            in_distribution=EVAL_IN_DIST, space="log1p", thresh=1.0, k_samples=_Kal,
        )
        print(f"[BS44] {_T} jours | indices biais: "
              + ", ".join(f"{k}={v:+.3f}" for k, v in _res44["indices"].items() if k.endswith("_bias")))
        print(f"[BS44] PSD distance = {_res44['psd_distance']:.5f}")
        print(f"💾 {_out44}")
