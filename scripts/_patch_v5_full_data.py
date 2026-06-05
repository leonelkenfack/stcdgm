"""Bootstrap autonome COMPLET : reprend exactement Cell 15+16+17 du training notebook
(stream_download + DATA_ROOT detection + BS32 SSD copy + Zenodo download HR/LR train + test files
+ gdown fallback Drive public pour statics/normalization), puis charge stacks et predict.
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

BACKUP = Path("st_cdgm_v5_evaluation.ipynb.bak7")
if not BACKUP.exists():
    BACKUP.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] Backup #7 cree : {BACKUP}")

FULL_BOOTSTRAP = '''# ============================================================
# Bootstrap autonome COMPLET (= Cells 15+16+17+30+32+40 du training notebook)
# Telecharge automatiquement TOUS les datasets manquants depuis Zenodo
# ============================================================
import os
import sys
import time
import json
import shutil
import urllib.request
import urllib.error
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

ON_COLAB = "google.colab" in sys.modules or Path("/content").exists()

# 1. CONFIG (base + override corrdiff_normal si dispo)
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

# 2. DATA_ROOT detection (= training cell 16)
DATA_ROOT_LOCAL = Path("data/raw")
DATA_ROOT_DRIVE = Path("/content/drive/MyDrive/climate_data/data")
_DATA_ROOT_LOCAL_SSD = Path("/content/data_local")

if ON_COLAB and DATA_ROOT_DRIVE.parent.parent.exists():
    DATA_ROOT = DATA_ROOT_DRIVE
    print(f"[INFO] DATA_ROOT = Drive ({DATA_ROOT})")
else:
    DATA_ROOT = DATA_ROOT_LOCAL
    print(f"[INFO] DATA_ROOT = local ({DATA_ROOT.resolve()})")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# 3. BS32 SSD copy (Drive -> /content/data_local pour I/O rapide)
_BS32_ENABLED = bool(globals().get("DATA_LOCAL_SSD", True))
if ON_COLAB and _BS32_ENABLED and DATA_ROOT == DATA_ROOT_DRIVE:
    _files_to_copy = [
        ("train/predictor_ACCESS-CM2_hist.nc",   "predictor_ACCESS-CM2_hist.nc"),
        ("train/pr_ACCESS-CM2_hist.nc",          "pr_ACCESS-CM2_hist.nc"),
        ("static_predictors/ERA5_eval_ccam_12km.198110_NZ_Invariant.nc",
         "ERA5_eval_ccam_12km.198110_NZ_Invariant.nc"),
        ("normalization_coefs/mean_1974_2011.nc", "mean_1974_2011.nc"),
        ("normalization_coefs/std_1974_2011.nc",  "std_1974_2011.nc"),
    ]
    _ssd_train = _DATA_ROOT_LOCAL_SSD / "train"
    _ssd_static = _DATA_ROOT_LOCAL_SSD / "static_predictors"
    _ssd_norm = _DATA_ROOT_LOCAL_SSD / "normalization_coefs"
    for _d in (_ssd_train, _ssd_static, _ssd_norm):
        _d.mkdir(parents=True, exist_ok=True)
    _t_total = time.time()
    _bytes_copied = 0
    for _rel, _name in _files_to_copy:
        _src = DATA_ROOT_DRIVE / _rel
        if "train/" in _rel:
            _dst = _ssd_train / _name
        elif "static_predictors/" in _rel:
            _dst = _ssd_static / _name
        else:
            _dst = _ssd_norm / _name
        if not _src.exists():
            continue
        if _dst.exists() and _dst.stat().st_size == _src.stat().st_size:
            continue
        _t0 = time.time()
        print(f"   copie {_src.name}...", flush=True)
        shutil.copy2(_src, _dst)
        _bytes_copied += _dst.stat().st_size
        print(f"   OK {_dst.name} ({_dst.stat().st_size/1e6:.0f} MB en {time.time()-_t0:.1f}s)")
    if _bytes_copied > 0:
        print(f"BS32 SSD copy: {_bytes_copied/1e9:.2f} GB en {time.time()-_t_total:.1f}s")
    DATA_ROOT = _DATA_ROOT_LOCAL_SSD
    print(f"[INFO] DATA_ROOT redirige vers SSD : {_DATA_ROOT_LOCAL_SSD}")

# 4. Resolution des paths (= training cell 16 suite)
def _relocate(p):
    if not p:
        return p
    s = str(p)
    if s.startswith("data/raw/"):
        return str(DATA_ROOT / s[len("data/raw/"):])
    return s

for _key in ("lr_path", "hr_path", "static_path"):
    if CONFIG.data.get(_key):
        CONFIG.data[_key] = _relocate(CONFIG.data[_key])

LR_PATH = str(CONFIG.data.lr_path)
HR_PATH = str(CONFIG.data.hr_path)
STATIC_PATH = str(CONFIG.data.static_path) if CONFIG.data.get("static_path") else None
MEAN_PATH = str(DATA_ROOT / "normalization_coefs" / "mean_1974_2011.nc")
STD_PATH = str(DATA_ROOT / "normalization_coefs" / "std_1974_2011.nc")

URL_ZENODO_HR = "https://zenodo.org/records/10889046/files/pr_ACCESS-CM2_hist.nc?download=1"
URL_ZENODO_LR = "https://zenodo.org/records/10889046/files/predictor_ACCESS-CM2_hist.nc?download=1"
URLS_TEST = [
    ("EC-Earth3_histupdated_compressed.nc",
     "https://zenodo.org/records/10889046/files/EC-Earth3_histupdated_compressed.nc?download=1"),
    ("EC-Earth3_historical_precip_compressed.nc",
     "https://zenodo.org/records/10889046/files/EC-Earth3_historical_precip_compressed.nc?download=1"),
    ("NorESM2-MM_histupdated_compressed.nc",
     "https://zenodo.org/records/10889046/files/NorESM2-MM_histupdated_compressed.nc?download=1"),
    ("NorESM2-MM_historical_precip_compressed.nc",
     "https://zenodo.org/records/10889046/files/NorESM2-MM_historical_precip_compressed.nc?download=1"),
]

# 5. stream_download (atomique + reprise + timeout)
def stream_download(url, dest, retries=5, chunk_size=1024*1024,
                    connect_timeout=30, read_timeout=120):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        already = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url)
        if already > 0:
            req.add_header("Range", f"bytes={already}-")
            print(f"   reprise a {already/1e6:.1f} MB")
        try:
            with urllib.request.urlopen(req, timeout=connect_timeout) as resp:
                total = resp.length
                if total is None and resp.headers.get("Content-Length"):
                    total = int(resp.headers["Content-Length"])
                grand_total = (total + already) if total else None
                mode = "ab" if already > 0 else "wb"
                with open(part, mode) as f:
                    downloaded = already
                    last_log = time.time()
                    last_log_bytes = downloaded
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if now - last_log >= 5.0:
                            speed = (downloaded - last_log_bytes) / (now - last_log) / 1e6
                            if grand_total:
                                pct = 100.0 * downloaded / grand_total
                                print(f"     {downloaded/1e6:7.1f}/{grand_total/1e6:7.1f} MB ({pct:.0f}%) {speed:.1f} MB/s")
                            else:
                                print(f"     {downloaded/1e6:7.1f} MB {speed:.1f} MB/s")
                            last_log = now
                            last_log_bytes = downloaded
            os.replace(part, dest)
            print(f"   OK {dest.name} ({dest.stat().st_size/1e6:.0f} MB)")
            return True
        except urllib.error.HTTPError as e:
            if e.code in (503, 504, 429):
                wait = min(60, 2**attempt); print(f"   HTTP {e.code} retry {wait}s"); time.sleep(wait)
            elif e.code == 416:
                os.replace(part, dest); return True
            else:
                print(f"   HTTP {e.code}: {e.reason}"); return False
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            wait = min(60, 2**attempt); print(f"   reseau retry {wait}s ({type(e).__name__})"); time.sleep(wait)
        except Exception as e:
            print(f"   ERREUR {type(e).__name__}: {e}"); return False
    return False

# 6. Telechargements train (HR, LR ACCESS-CM2)
if not Path(HR_PATH).exists():
    print(f"Download HR ACCESS-CM2: {HR_PATH}")
    if not stream_download(URL_ZENODO_HR, HR_PATH):
        raise RuntimeError("Echec download HR ACCESS-CM2")
if not Path(LR_PATH).exists():
    print(f"Download LR ACCESS-CM2: {LR_PATH}")
    if not stream_download(URL_ZENODO_LR, LR_PATH):
        raise RuntimeError("Echec download LR ACCESS-CM2")

# 7. Telechargements test (EC-Earth3, NorESM2-MM)
TEST_ROOT = DATA_ROOT / "test"
TEST_ROOT.mkdir(parents=True, exist_ok=True)
for _filename, _url in URLS_TEST:
    _filepath = TEST_ROOT / _filename
    if _filepath.exists():
        continue
    print(f"Download test: {_filename}")
    if not stream_download(_url, str(_filepath)):
        raise RuntimeError(f"Echec download {_filename}")

# 8. Statics + normalization (fallback gdown si Drive public)
_PUBLIC_DRIVE_FALLBACKS = {
    "static_predictors/ERA5_eval_ccam_12km.198110_NZ_Invariant.nc":
        "1KY6IS1W5Wt-l_xyV7Qw8caA49zPzuSEx",
    "normalization_coefs/mean_1974_2011.nc":
        "14wVaJTUDgLwLlFcqRFA6pzJg9tZtAVQ0",
    "normalization_coefs/std_1974_2011.nc":
        "1ycqq9DqpfdOOiyQqgKs797OzRdHND3ZL",
}

def _gdown_install():
    try:
        import gdown; return True
    except ImportError:
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"], timeout=120)
            import gdown; return True
        except Exception:
            return False

def _try_gdown(path):
    if not path:
        return False
    pth = Path(path)
    rel_key = None
    for _key in _PUBLIC_DRIVE_FALLBACKS:
        if str(pth).endswith(_key.replace("/", os.sep)) or str(pth).endswith(_key):
            rel_key = _key; break
    if rel_key is None:
        return False
    file_id = _PUBLIC_DRIVE_FALLBACKS[rel_key]
    pth.parent.mkdir(parents=True, exist_ok=True)
    if not _gdown_install():
        return False
    import gdown
    try:
        print(f"   gdown.download(id={file_id}) -> {pth}")
        gdown.download(id=file_id, output=str(pth), quiet=False)
        return pth.exists() and pth.stat().st_size > 0
    except Exception as e:
        print(f"   gdown ERREUR: {e}"); return False

for _var, _name in [("STATIC_PATH", "Static"), ("MEAN_PATH", "Mean"), ("STD_PATH", "Std")]:
    _p = globals()[_var]
    if _p and Path(_p).exists():
        continue
    if _try_gdown(_p):
        print(f"   OK {_name} (gdown public)")
    else:
        print(f"   {_name} absent: {_p} -> None")
        globals()[_var] = None

# 9. Resume final
print()
print("Datasets disponibles :")
print(f"  LR train  : {LR_PATH}  ({'OK' if Path(LR_PATH).exists() else 'MISSING'})")
print(f"  HR train  : {HR_PATH}  ({'OK' if Path(HR_PATH).exists() else 'MISSING'})")
print(f"  Static    : {STATIC_PATH}  ({'OK' if STATIC_PATH and Path(STATIC_PATH).exists() else 'NONE'})")
print(f"  Mean/Std  : {MEAN_PATH} / {STD_PATH}")
for fname, _ in URLS_TEST:
    p = TEST_ROOT / fname
    print(f"  Test      : {p.name}  ({'OK' if p.exists() else 'MISSING'})")

# 10. GCM_REGISTRY pour OOD
GCM_REGISTRY = {
    "ACCESS-CM2":  (Path(LR_PATH), Path(HR_PATH), True),
    "EC-Earth3":   (TEST_ROOT / "EC-Earth3_histupdated_compressed.nc",
                     TEST_ROOT / "EC-Earth3_historical_precip_compressed.nc", False),
    "NorESM2-MM":  (TEST_ROOT / "NorESM2-MM_histupdated_compressed.nc",
                     TEST_ROOT / "NorESM2-MM_historical_precip_compressed.nc", False),
}

# 11. Pipeline + builder
from st_cdgm.data.pipeline import NetCDFDataPipeline
from st_cdgm.models.graph_builder import HeteroGraphBuilder

def make_pipeline(lr_path, hr_path):
    return NetCDFDataPipeline(
        lr_path=str(lr_path), hr_path=str(hr_path),
        static_path=str(STATIC_PATH) if STATIC_PATH and Path(STATIC_PATH).exists() else None,
        seq_len=int(CONFIG.data.seq_len),
        baseline_strategy=str(CONFIG.data.baseline_strategy),
        baseline_factor=int(CONFIG.data.baseline_factor),
        target_transform=str(CONFIG.data.get("target_transform", "log1p")),
        normalize=bool(CONFIG.data.normalize),
        nan_fill_strategy=str(CONFIG.data.nan_fill_strategy),
        precipitation_delta=float(CONFIG.data.get("precipitation_delta", 0.01)),
        lr_variables=list(CONFIG.data.lr_variables),
        hr_variables=list(CONFIG.data.hr_variables),
        static_variables=list(CONFIG.data.static_variables) if STATIC_PATH and Path(STATIC_PATH).exists() else None,
        means_path=str(MEAN_PATH) if MEAN_PATH and Path(MEAN_PATH).exists() else None,
        stds_path=str(STD_PATH) if STD_PATH and Path(STD_PATH).exists() else None,
        eager_load_datasets=bool(CONFIG.data.get("eager_load_datasets", False)),
    )

pipeline_access = make_pipeline(LR_PATH, HR_PATH)
builder = HeteroGraphBuilder(
    lr_shape=lr_shape, hr_shape=hr_shape,
    static_dataset=pipeline_access.get_static_dataset(),
    include_mid_layer=bool(CONFIG.graph.include_mid_layer),
)
print()
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
    print(f"[INFO] CONFIG.rcn.driver_dim -> {_runtime_dim}")

# 12. Stacks
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

# 13. Predict generique
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
print("=" * 70)
print("Bootstrap autonome COMPLET")
print("=" * 70)
print("Variables disponibles :")
print(f"  CONFIG, DEVICE, builder, convert_sample_to_batch, predict_with_stack")
print(f"  stack_v5, stack_nc, GCM_REGISTRY, make_pipeline")
print(f"  test_dataset (ACCESS-CM2 in-dist)")
print()
print(f"DATA_ROOT  : {DATA_ROOT}")
print(f"TEST_ROOT  : {TEST_ROOT}")
'''

nb["cells"][4]["source"] = [ln + "\n" for ln in FULL_BOOTSTRAP.split("\n")[:-1]] + [FULL_BOOTSTRAP.split("\n")[-1]]

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[OK] Cell 4 mise a jour (bootstrap autonome COMPLET)")
print(f"     Logique Cell 16+17 du training notebook integree")
print(f"     {len(nb['cells'])} cellules au total")
