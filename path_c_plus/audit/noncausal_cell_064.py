# >>> BS45 — DATALOADER CONTINU PAR GCM (in-dist ACCESS-CM2 / OOD EC-Earth3, NorESM2)
# Construit un dataloader ORDONNE (stride=1, sans shuffle, pleine periode) pour un
# GCM donne, en repliquant EXACTEMENT les args du pipeline d'entrainement (memes
# variables, normalisation, baseline, seq_len). Les fichiers OOD sont drop-in :
# memes 15 variables LR (t/u/v/w/q x 850/500/250) + meme grille (LR 23x26, HR 172x179).
# Pose ALIGNED_EVAL_LOADER + EVAL_GCM_TAG + EVAL_IN_DIST, consommes par BS44.
from st_cdgm.data.pipeline import NetCDFDataPipeline
from torch.utils.data import DataLoader as _DL45, IterableDataset as _IDS45
from pathlib import Path as _P45

_root45 = _P45(str(globals().get("DATA_ROOT", "data/raw")))

# Registre GCM -> (chemin LR, chemin HR relatif a DATA_ROOT, in_distribution).
GCM_REGISTRY = {
    "ACCESS-CM2":  ("train/predictor_ACCESS-CM2_hist.nc",          "train/pr_ACCESS-CM2_hist.nc",          True),
    "EC-Earth3":   ("test/EC-Earth3_histupdated_compressed.nc",    "test/EC-Earth3_historical_precip_compressed.nc",  False),
    "NorESM2-MM":  ("test/NorESM2-MM_histupdated_compressed.nc",   "test/NorESM2-MM_historical_precip_compressed.nc", False),
}

# GCM cible : definir EVAL_GCM_TAG avant cette cellule (defaut ACCESS-CM2 in-dist).
EVAL_GCM_TAG = str(globals().get("EVAL_GCM_TAG", "ACCESS-CM2"))
if EVAL_GCM_TAG not in GCM_REGISTRY:
    raise ValueError(f"EVAL_GCM_TAG={EVAL_GCM_TAG!r} inconnu. Choix: {list(GCM_REGISTRY)}")
_lr_rel, _hr_rel, _in_dist = GCM_REGISTRY[EVAL_GCM_TAG]
EVAL_IN_DIST = bool(_in_dist)
_lr45, _hr45 = str(_root45 / _lr_rel), str(_root45 / _hr_rel)
print(f"[BS45] GCM={EVAL_GCM_TAG} | in_dist={EVAL_IN_DIST}")
print(f"       LR={_lr45}\n       HR={_hr45}")

# Pipeline GCM : MEMES args que l'entrainement (on ne change QUE les chemins).
_static45 = str(globals().get("STATIC_PATH", "")) or None
_mean45 = globals().get("MEAN_PATH", None)
_std45 = globals().get("STD_PATH", None)
import os as _os45
_pipe45 = NetCDFDataPipeline(
    lr_path=_lr45,
    hr_path=_hr45,
    static_path=_static45,
    seq_len=int(globals().get("SEQ_LEN", 16)),
    baseline_strategy=str(globals().get("BASELINE_STRATEGY", "hr_smoothing")),
    baseline_factor=int(globals().get("BASELINE_FACTOR", 4)),
    normalize=bool(globals().get("NORMALIZE", False)),
    nan_fill_strategy=str(globals().get("NAN_FILL_STRATEGY", "zero")),
    precipitation_delta=float(globals().get("PRECIPITATION_DELTA", 0.01)),
    lr_variables=list(globals().get("LR_VARIABLES")) if globals().get("LR_VARIABLES") else None,
    hr_variables=list(globals().get("HR_VARIABLES")) if globals().get("HR_VARIABLES") else None,
    static_variables=globals().get("STATIC_VARIABLES", None),
    means_path=_mean45 if (_mean45 and _os45.path.exists(str(_mean45))) else None,
    stds_path=_std45 if (_std45 and _os45.path.exists(str(_std45))) else None,
)

# Dataset ORDONNE : stride=1 (serie journaliere continue -> CDD/Rx1Day corrects),
# training=False (aucune augmentation), drop_last pour des fenetres pleines.
_ds45 = _pipe45.build_sequence_dataset(
    seq_len=int(globals().get("SEQ_LEN", 16)),
    stride=1, drop_last=True, as_torch=True, training=False,
)

# DataLoader sans shuffle (IterableDataset = ordre du generateur = ordre temporel).
_dl_kwargs45 = dict(batch_size=int(globals().get("BATCH_SIZE", 64)),
                    num_workers=int(globals().get("NUM_WORKERS", 0)),
                    pin_memory=bool(globals().get("PIN_MEMORY", False)),
                    collate_fn=lambda x: x)
if not isinstance(_ds45, _IDS45):
    _dl_kwargs45["shuffle"] = False
ALIGNED_EVAL_LOADER = _DL45(_ds45, **_dl_kwargs45)
print(f"[BS45] ALIGNED_EVAL_LOADER pret (stride=1, ordonne). "
      f"Execute BS44 ensuite pour produire aligned_metrics_{EVAL_GCM_TAG}_<variant>.json.")
print("[BS45] Pour l'OOD : redefinir EVAL_GCM_TAG='EC-Earth3' (ou 'NorESM2-MM') "
      "puis re-executer BS45 + BS44.")
