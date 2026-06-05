"""Patch bootstrap pour qu'il :
- essaye plusieurs chemins DATA_ROOT candidats
- telecharge automatiquement les datasets test depuis Zenodo si absents
- echoue clairement si ACCESS-CM2 introuvable (non-Zenodo)
"""
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

NB = Path("st_cdgm_v5_evaluation.ipynb")
with NB.open(encoding="utf-8") as f:
    nb = json.load(f)

BACKUP = Path("st_cdgm_v5_evaluation.ipynb.bak6")
if not BACKUP.exists():
    BACKUP.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] Backup #6 cree : {BACKUP}")

NEW_BOOTSTRAP = '''# ============================================================
# Bootstrap autonome — robuste sur paths + auto-download Zenodo
# ============================================================
import os
import sys
import time
import json
import urllib.request
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

ON_COLAB = "google.colab" in sys.modules or Path("/content").exists()

# 1. CONFIG
_base = Path("config/training_config.yaml")
_override = Path("config/training_config_corrdiff_normal.yaml")
CONFIG = OmegaConf.load(_base)
if _override.exists():
    CONFIG = OmegaConf.merge(CONFIG, OmegaConf.load(_override))
    print("[OK] CONFIG = base + corrdiff_normal override")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr_shape = tuple(CONFIG.graph.lr_shape)
hr_shape = tuple(CONFIG.graph.hr_shape)
print(f"[OK] DEVICE={DEVICE}  lr_shape={lr_shape}  hr_shape={hr_shape}")

# 2. Recherche DATA_ROOT (multiples candidats)
CANDIDATES = [
    Path("/content/data_local"),                          # SSD copy (training notebook BS32)
    Path("/content/drive/MyDrive/climate_data/data/raw"), # Drive standard
    Path("/content/drive/MyDrive/climate_data/data"),     # Drive sans /raw
    Path("/content/climate_data/data/raw"),               # Repo cloné
    Path("data/raw"),                                     # Local relatif
]

def find_file(filename_rel):
    """Cherche un fichier dans tous les candidats. Retourne le premier path existant."""
    for root in CANDIDATES:
        p = root / filename_rel
        if p.exists():
            return p
    return None

# 3. Resolution des paths concrets
print()
print("Recherche des datasets dans les chemins candidats...")
for c in CANDIDATES:
    exists = "OK" if c.exists() else "--"
    print(f"  [{exists}] {c}")
print()

LR_PATH_ACCESS  = find_file("train/predictor_ACCESS-CM2_hist.nc")
HR_PATH_ACCESS  = find_file("train/pr_ACCESS-CM2_hist.nc")
STATIC_PATH     = find_file("static_predictors/ERA5_eval_ccam_12km.198110_NZ_Invariant.nc")
MEAN_PATH       = find_file("normalization_coefs/mean_1974_2011.nc")
STD_PATH        = find_file("normalization_coefs/std_1974_2011.nc")

LR_PATH_ECEARTH = find_file("test/EC-Earth3_histupdated_compressed.nc")
HR_PATH_ECEARTH = find_file("test/EC-Earth3_historical_precip_compressed.nc")
LR_PATH_NORESM  = find_file("test/NorESM2-MM_histupdated_compressed.nc")
HR_PATH_NORESM  = find_file("test/NorESM2-MM_historical_precip_compressed.nc")

# 4. Telechargement automatique des datasets test si absents (Zenodo)
ZENODO_URLS = {
    "test/EC-Earth3_histupdated_compressed.nc":
        "https://zenodo.org/records/10889046/files/EC-Earth3_histupdated_compressed.nc?download=1",
    "test/EC-Earth3_historical_precip_compressed.nc":
        "https://zenodo.org/records/10889046/files/EC-Earth3_historical_precip_compressed.nc?download=1",
    "test/NorESM2-MM_histupdated_compressed.nc":
        "https://zenodo.org/records/10889046/files/NorESM2-MM_histupdated_compressed.nc?download=1",
    "test/NorESM2-MM_historical_precip_compressed.nc":
        "https://zenodo.org/records/10889046/files/NorESM2-MM_historical_precip_compressed.nc?download=1",
}

# Determine ou ecrire les telechargements : prefere Drive si dispo, sinon SSD
def pick_download_root():
    drive_data = Path("/content/drive/MyDrive/climate_data/data")
    if drive_data.parent.exists():
        return drive_data
    return Path("/content/data_local")

def stream_download(url, dst, max_retries=4):
    """Download avec retry + reprise (range header)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    if dst.exists():
        existing = dst.stat().st_size
    else:
        existing = 0
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            if existing > 0:
                req.add_header("Range", f"bytes={existing}-")
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = int(resp.headers.get("Content-Length", 0)) + existing
                mode = "ab" if existing > 0 else "wb"
                with open(dst, mode) as f:
                    chunk_size = 1024 * 1024
                    downloaded = existing
                    last_print = time.time()
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if time.time() - last_print > 5:
                            pct = 100 * downloaded / max(total, 1)
                            print(f"   {dst.name} {downloaded/1e6:.0f}/{total/1e6:.0f} MB ({pct:.0f}%)")
                            last_print = time.time()
            return True
        except Exception as e:
            print(f"   reseau ({type(e).__name__}: {e}) — retry dans {2**(attempt+1)}s")
            time.sleep(2**(attempt+1))
            if dst.exists():
                existing = dst.stat().st_size
    return False

# Telecharge les test files manquants
missing_test = []
for rel in ZENODO_URLS:
    if find_file(rel) is None:
        missing_test.append(rel)

if missing_test:
    download_root = pick_download_root()
    print(f"Telechargement Zenodo vers {download_root} (test files manquants : {len(missing_test)})")
    for rel in missing_test:
        dst = download_root / rel
        if dst.exists():
            continue
        print(f"  Download {rel}...")
        ok = stream_download(ZENODO_URLS[rel], dst)
        if ok:
            print(f"  [OK] {dst.name} ({dst.stat().st_size/1e6:.0f} MB)")
        else:
            print(f"  [FAIL] {dst.name}")
    # Re-resolve apres download
    LR_PATH_ECEARTH = find_file("test/EC-Earth3_histupdated_compressed.nc")
    HR_PATH_ECEARTH = find_file("test/EC-Earth3_historical_precip_compressed.nc")
    LR_PATH_NORESM  = find_file("test/NorESM2-MM_histupdated_compressed.nc")
    HR_PATH_NORESM  = find_file("test/NorESM2-MM_historical_precip_compressed.nc")
    print()

# 5. Verification des paths critiques
print("Resolution finale :")
for label, p in [
    ("ACCESS-CM2 LR (train)", LR_PATH_ACCESS),
    ("ACCESS-CM2 HR (train)", HR_PATH_ACCESS),
    ("Static", STATIC_PATH), ("Mean", MEAN_PATH), ("Std", STD_PATH),
    ("EC-Earth3 LR (OOD)", LR_PATH_ECEARTH),
    ("EC-Earth3 HR (OOD)", HR_PATH_ECEARTH),
    ("NorESM2-MM LR (OOD)", LR_PATH_NORESM),
    ("NorESM2-MM HR (OOD)", HR_PATH_NORESM),
]:
    status = "OK" if (p is not None and p.exists()) else "MISSING"
    print(f"  [{status:7s}] {label:<26s} {p}")

# Erreur claire si fichiers training manquants (pas dans Zenodo)
if LR_PATH_ACCESS is None or HR_PATH_ACCESS is None:
    print()
    print("=" * 70)
    print("ERREUR : Fichiers ACCESS-CM2 (train) introuvables.")
    print("=" * 70)
    print("Le fichier 'predictor_ACCESS-CM2_hist.nc' et 'pr_ACCESS-CM2_hist.nc'")
    print("ne sont pas dans Zenodo public — il faut les obtenir autrement.")
    print()
    print("Solutions :")
    print("  1. Lancer une fois le notebook training (Cellule 16 BS32) qui copie")
    print("     les fichiers de Drive vers SSD /content/data_local/")
    print("  2. Ou uploader directement les fichiers a un de ces chemins :")
    for c in CANDIDATES:
        print(f"     - {c}/train/")
    print()
    print("Alternative : pour valider l'OOD seul (sans in-distribution ACCESS),")
    print("on peut utiliser EC-Earth3 comme baseline a la place. Mais Phase 6")
    print("requiert ACCESS-CM2.")
    raise FileNotFoundError("ACCESS-CM2 training data manquantes")

GCM_REGISTRY = {
    "ACCESS-CM2":  (LR_PATH_ACCESS,  HR_PATH_ACCESS,  True),
    "EC-Earth3":   (LR_PATH_ECEARTH, HR_PATH_ECEARTH, False),
    "NorESM2-MM":  (LR_PATH_NORESM,  HR_PATH_NORESM,  False),
}

# 6. Pipeline + builder
from st_cdgm.data.pipeline import NetCDFDataPipeline
from st_cdgm.models.graph_builder import HeteroGraphBuilder

def make_pipeline(lr_path, hr_path):
    return NetCDFDataPipeline(
        lr_path=str(lr_path), hr_path=str(hr_path),
        static_path=str(STATIC_PATH) if STATIC_PATH and STATIC_PATH.exists() else None,
        seq_len=int(CONFIG.data.seq_len),
        baseline_strategy=str(CONFIG.data.baseline_strategy),
        baseline_factor=int(CONFIG.data.baseline_factor),
        target_transform=str(CONFIG.data.get("target_transform", "log1p")),
        normalize=bool(CONFIG.data.normalize),
        nan_fill_strategy=str(CONFIG.data.nan_fill_strategy),
        precipitation_delta=float(CONFIG.data.get("precipitation_delta", 0.01)),
        lr_variables=list(CONFIG.data.lr_variables),
        hr_variables=list(CONFIG.data.hr_variables),
        static_variables=list(CONFIG.data.static_variables) if STATIC_PATH and STATIC_PATH.exists() else None,
        means_path=str(MEAN_PATH) if MEAN_PATH and MEAN_PATH.exists() else None,
        stds_path=str(STD_PATH) if STD_PATH and STD_PATH.exists() else None,
        eager_load_datasets=bool(CONFIG.data.get("eager_load_datasets", False)),
    )

pipeline_access = make_pipeline(LR_PATH_ACCESS, HR_PATH_ACCESS)
builder = HeteroGraphBuilder(
    lr_shape=lr_shape, hr_shape=hr_shape,
    static_dataset=pipeline_access.get_static_dataset(),
    include_mid_layer=bool(CONFIG.graph.include_mid_layer),
)
print(f"[OK] Builder cree ({len(builder.dynamic_node_types)} dyn + {len(builder.static_node_types)} static)")

def convert_sample_to_batch(sample, builder, device):
    lr_seq = sample["lr"]
    seq_len = lr_seq.shape[0]
    lr_nodes_steps = [builder.lr_grid_to_nodes(lr_seq[t]) for t in range(seq_len)]
    lr_tensor = torch.stack(lr_nodes_steps, dim=0)
    dynamic_features = {nt: lr_nodes_steps[0] for nt in builder.dynamic_node_types}
    hetero = builder.prepare_step_data(dynamic_features).to(device)
    return {"lr": lr_tensor, "residual": sample["residual"],
            "baseline": sample.get("baseline"), "hetero": hetero}

test_dataset = pipeline_access.build_sequence_dataset(
    seq_len=int(CONFIG.data.seq_len), stride=int(CONFIG.data.stride), as_torch=True,
)
sample = next(iter(test_dataset))
_runtime_dim = int(sample["lr"].shape[1])
if _runtime_dim != int(CONFIG.rcn.driver_dim):
    CONFIG.rcn.driver_dim = _runtime_dim
    CONFIG.rcn.reconstruction_dim = _runtime_dim
    print(f"[INFO] CONFIG.rcn.driver_dim aligne sur runtime: {_runtime_dim}")

# 7. Stacks
from st_cdgm.models import (
    IntelligibleVariableEncoder, IntelligibleVariableConfig,
    GraphToGridDecoder, RCNCell, RCNSequenceRunner,
    CausalDiffusionDecoder,
)
from st_cdgm.models.edm_preconditioner import EDMConfig
try:
    from st_cdgm.models import ConditionalSkipBlock
    SKIP_AVAILABLE = True
except ImportError:
    SKIP_AVAILABLE = False
    ConditionalSkipBlock = None

def build_stack(ckpt_path, name):
    print(f"  [{name}] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    enc_cfg = IntelligibleVariableConfig(
        num_variables=len(CONFIG.data.lr_variables),
        hidden_dim=CONFIG.encoder.hidden_dim,
        conditioning_dim=CONFIG.encoder.conditioning_dim,
        num_dag_tokens=int(CONFIG.encoder.get("num_dag_tokens", 2)),
        causal_conditioning=bool(CONFIG.encoder.get("causal_conditioning", True)),
    )
    enc = IntelligibleVariableEncoder(enc_cfg).to(DEVICE)
    rcn_cell = RCNCell(
        num_variables=enc_cfg.num_variables,
        hidden_dim=CONFIG.rcn.hidden_dim,
        driver_dim=CONFIG.rcn.driver_dim,
        reconstruction_dim=CONFIG.rcn.reconstruction_dim,
        dropout=CONFIG.rcn.dropout,
    ).to(DEVICE)
    rcn_runner = RCNSequenceRunner(rcn_cell, detach_interval=CONFIG.rcn.detach_interval)
    rh = GraphToGridDecoder(
        d_model=CONFIG.encoder.hidden_dim,
        hr_h=CONFIG.graph.hr_shape[0], hr_w=CONFIG.graph.hr_shape[1],
    ).to(DEVICE)
    edm_cfg = EDMConfig.from_yaml_dict(CONFIG.diffusion.get("edm", {}))
    diff = CausalDiffusionDecoder(
        in_channels=CONFIG.diffusion.in_channels,
        conditioning_dim=CONFIG.diffusion.conditioning_dim,
        height=CONFIG.diffusion.height, width=CONFIG.diffusion.width,
        scheduler_type=CONFIG.diffusion.scheduler_type, causal_concat=True,
        edm_config=edm_cfg, unet_kwargs=dict(CONFIG.diffusion.unet_kwargs),
    ).to(DEVICE)
    for n, m in [("encoder", enc), ("rcn_cell", rcn_cell),
                  ("regression_head", rh), ("diffusion", diff)]:
        k = f"{n}_state_dict"
        if k in ckpt:
            m.load_state_dict(ckpt[k])
    skip = None
    if SKIP_AVAILABLE and "skip_block_state_dict" in ckpt:
        skip = ConditionalSkipBlock(
            lr_channels=len(CONFIG.data.lr_variables),
            hr_shape=tuple(CONFIG.graph.hr_shape),
        ).to(DEVICE)
        skip.load_state_dict(ckpt["skip_block_state_dict"])
        print(f"  [{name}] [+] skip_block ({skip.num_params()} params)")
    enc.eval(); rcn_cell.eval(); rh.eval(); diff.eval()
    if skip is not None:
        skip.eval()
    A_dag = rcn_cell.A_dag.detach().cpu().clone() if hasattr(rcn_cell, "A_dag") else None
    return {"encoder": enc, "rcn_runner": rcn_runner, "regression_head": rh,
            "diffusion": diff, "skip_block": skip, "A_dag": A_dag, "variant": name}

print()
print("Chargement des stacks...")
t0 = time.time()
stack_v5 = build_stack(V5_DIR / "epoch_last.pth", "V5")
stack_nc = build_stack(NONCAUSAL_DIR / "epoch_last.pth", "Noncausal")
print(f"[OK] 2 stacks charges en {time.time()-t0:.1f}s")

# 8. Predict generique
@torch.no_grad()
def predict_with_stack(stack, batch, K=4, n_steps=32):
    enc, rcn, rh, diff, skip = (stack["encoder"], stack["rcn_runner"],
                                  stack["regression_head"], stack["diffusion"],
                                  stack["skip_block"])
    lr = batch["lr"].to(DEVICE)
    H_init = enc.init_state(batch["hetero"]).to(DEVICE)
    drivers = [lr[t] for t in range(lr.shape[0])]
    seq = rcn.run(H_init, drivers, reconstruction_sources=None)
    H_T = seq.states[-1]
    mu_c = rh(H_T)
    tshape = batch["residual"][-1].to(DEVICE).shape
    if tshape[-2:] != mu_c.shape[-2:]:
        mu_c = torch.nn.functional.interpolate(
            mu_c, size=tshape[-2:], mode="bilinear", align_corners=False,
        )
    if skip is not None:
        lr_last = drivers[-1] if drivers[-1].dim() == 4 else drivers[-1].unsqueeze(0)
        mu, _ = skip(lr_last, mu_c)
    else:
        mu = mu_c
    mu = torch.nan_to_num(mu, nan=0.0)
    bl = batch["baseline"][-1].to(DEVICE)
    if bl.dim() == mu.dim() - 1:
        bl = bl.unsqueeze(0)
    bl = torch.nan_to_num(bl, nan=0.0)
    ens = []
    for _ in range(K):
        o = diff.sample(
            conditioning=None, num_steps=n_steps,
            scheduler_type="edm_karras", apply_constraints=False,
            mu_HR=mu, baseline_log=bl,
        )
        r = o.residual if hasattr(o, "residual") else o
        ens.append((bl + mu + r).cpu())
    return torch.stack(ens, dim=0)

print()
print("Bootstrap autonome complet. Variables disponibles :")
print(f"  CONFIG, DEVICE, builder, convert_sample_to_batch, predict_with_stack")
print(f"  stack_v5, stack_nc, GCM_REGISTRY, make_pipeline")
print(f"  test_dataset (ACCESS-CM2 in-dist)")
'''

# Replace cell 4 (bootstrap code)
nb["cells"][4]["source"] = [ln + "\n" for ln in NEW_BOOTSTRAP.split("\n")[:-1]] + [NEW_BOOTSTRAP.split("\n")[-1]]

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[OK] Cell 4 (bootstrap) mise a jour : {NB}")
print(f"     {len(nb['cells'])} cellules au total")
