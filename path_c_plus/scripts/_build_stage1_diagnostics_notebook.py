"""Génère `st_cdgm_stage1_diagnostics.ipynb` — les diagnostics étage 1 qui
EXIGENT le modèle entraîné (donc non exécutables hors Colab/GPU).

Complète les diagnostics indépendants du modèle déjà exécutés sur CPU
(`scripts/diag_stage1.py`, `diag_stage1b.py` → `results/diag_stage1*.json`,
synthèse au §9 de `docs/architecture_v8_design.md`).

Cinq diagnostics :
  A  Prédictibilité résiduelle  — LE test décisif : l'étage 1 détruit-il
     de l'information récupérable, ou son lissage EST-il la vraie moyenne
     conditionnelle ? (R² hors échantillon de g(y) -> r)
  B  Histogramme de alpha conditionné à l'intensité — la pénalité porte sur
     la MOYENNE du batch : une distribution bimodale la satisfait tout en
     produisant la régression des extrêmes.
  C  RAPSD de mu_causal seul — déficit HF PLAT (≈ alpha², le skip block est
     coupable) vs CROISSANT avec k (le décodeur est coupable).
  D  F1@p99 par membre vs sur la moyenne d'ensemble — la métrique est-elle
     mesurée sur un champ lissé deux fois ?
  E  Biais de Jensen du modèle réel — confirme sur le modèle les 10-18 %
     mesurés sur données seules.

Sortie : path_c_plus/scripts/st_cdgm_stage1_diagnostics.ipynb
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "path_c_plus" / "scripts" / "st_cdgm_stage1_diagnostics.ipynb"


def md(t: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.splitlines(keepends=True)}


C0 = """# Diagnostics étage 1 — Oracle / ST-CDGM

**Objet** : trancher *pourquoi* l'étage 1 lisse, et *si* c'est un défaut.

Les diagnostics indépendants du modèle ont déjà été exécutés sur CPU
(`results/diag_stage1.json`, `diag_stage1b.json`, synthèse §9 de
`docs/architecture_v8_design.md`). Ils ont établi :

| Résultat | Statut |
|---|---|
| Biais de Jensen (log1p → mm) : **10,6 % à 1 mm/j, 18,5 % sur le bin le plus intense** | CONFIRMÉ |
| Le `ConditionalSkipBlock` **n'explique pas** la régression F1@p99 (à 48 km, α n'a aucun effet) | RÉFUTÉ |
| Prédictibilité jusqu'à **9–13 km**, décodeur bloqué à **48 km** (grille 43×45) | CONFIRMÉ |
| Seulement **1 %** de la variance HR sous 48 km → la MSE est aveugle aux extrêmes | CONFIRMÉ |
| Le biais CDD ne vient **pas** du seuillage de la moyenne conditionnelle | RÉFUTÉ |

Ce notebook exécute les **cinq diagnostics restants**, qui exigent le modèle entraîné.

> ⚠️ **NorESM2-MM est un holdout pré-enregistré. Ce notebook ne doit JAMAIS l'ouvrir.**
> Une assertion le vérifie en Cell 2.
"""

C1 = """# >>> Cell 1 : Bootstrap Colab + git sync
import os, sys, subprocess
from pathlib import Path

GIT_URL    = "https://github.com/leonelkenfack/stcdgm.git"
GIT_BRANCH = "four-node-causal"
REPO_DIR   = "/content/climate_data"
DRIVE_ROOT = "/content/drive/MyDrive/climate_data"

if not Path("/content/drive").exists():
    from google.colab import drive
    drive.mount("/content/drive")

if not Path(REPO_DIR).exists():
    subprocess.check_call(["git", "clone", "--depth=200", "-b", GIT_BRANCH, GIT_URL, REPO_DIR])
else:
    subprocess.check_call(["git", "-C", REPO_DIR, "fetch", "origin"])
    subprocess.check_call(["git", "-C", REPO_DIR, "checkout", GIT_BRANCH])
    subprocess.check_call(["git", "-C", REPO_DIR, "pull", "origin", GIT_BRANCH])

for _p in (REPO_DIR, str(Path(REPO_DIR) / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(REPO_DIR)

try:
    import torch_geometric, cftime, h5netcdf, xbatcher, diffusers, omegaconf  # noqa: F401
    print("[Cell 1] deps critiques OK — pip install sauté.")
except Exception as _e:
    print(f"[Cell 1] pip install requis : {_e}")
    _EXTRA = ["omegaconf==2.3.0", "hydra-core==1.3.2", "diffusers==0.36.0",
              "transformers==4.57.6", "accelerate==1.12.0", "huggingface-hub==0.36.0",
              "safetensors==0.7.0", "xbatcher", "webdataset", "cftime", "h5netcdf",
              "netcdf4", "numcodecs", "scipy", "torch-geometric", "xformers"]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "--no-warn-script-location", *_EXTRA])

_sha = subprocess.check_output(["git", "-C", REPO_DIR, "rev-parse", "HEAD"]).decode().strip()
print(f"[Cell 1] OK — commit {_sha[:8]}")
"""

C2 = """# >>> Cell 2 : Checkpoint + config (lue DANS le checkpoint) + garde holdout
import torch, numpy as np, json, warnings
from omegaconf import OmegaConf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)

# ---- À RENSEIGNER : le checkpoint étage 1 à diagnostiquer -------------------
CKPT_PATH = f"{DRIVE_ROOT}/checkpoints/st_cdgm_checkpoint.pth"   # <— ADAPTER
N_DAYS    = 365      # jours évalués (365 = 1 an, suffisant et rapide)
K_MEMBERS = 12       # membres pour le diagnostic D (F1 par membre)
SEED      = 42
# ---------------------------------------------------------------------------

assert Path(CKPT_PATH).exists(), f"checkpoint absent : {CKPT_PATH}"
ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
print("clés du checkpoint :", [k for k in ck.keys()][:12])

# La config est stockée DANS le checkpoint -> pas de dérive de configuration.
if "config_full" in ck:
    CONFIG = OmegaConf.create(ck["config_full"])
elif "config" in ck:
    CONFIG = OmegaConf.create(ck["config"])
else:
    CONFIG = OmegaConf.load("config/training_config.yaml")
    warnings.warn("config absente du checkpoint — fallback sur le yaml du repo")

HAS_SKIP = any("skip" in k.lower() for k in ck.keys())
HAS_RH   = any("regression_head" in k for k in ck.keys())
print(f"skip_block dans le ckpt : {HAS_SKIP} | regression_head : {HAS_RH}")
print(f"grille intermédiaire décodeur : "
      f"{CONFIG.two_stage.regression_head.intermediate_h}x"
      f"{CONFIG.two_stage.regression_head.intermediate_w}")

# ---- GARDE-FOU HOLDOUT : NorESM2-MM ne doit jamais être ouvert -------------
FORBIDDEN = "NorESM2"
def _guard(path: str):
    assert FORBIDDEN not in str(path), (
        f"STOP — {path} touche le holdout pré-enregistré NorESM2-MM. "
        "Le consommer détruit la seule preuve OOD du projet.")
    return path
print("[Cell 2] garde-fou holdout actif")
"""

C3 = """# >>> Cell 3 : Pipeline + reconstruction de la pile + chargement des poids
from st_cdgm.data.pipeline import NetCDFDataPipeline
from st_cdgm.models.graph_builder import HeteroGraphBuilder
from st_cdgm.models.intelligible_encoder import IntelligibleVariableEncoder, IntelligibleVariableConfig
from st_cdgm.models.causal_rcn import RCNCell, RCNSequenceRunner
from st_cdgm.models.regression_head import GraphToGridDecoder

GCM_ID   = "ACCESS-CM2"                      # in-distribution uniquement
LR_PATH  = _guard(f"{DRIVE_ROOT}/lr_{GCM_ID}_v6.nc")
HR_PATH  = _guard(f"{DRIVE_ROOT}/data/train/pr_{GCM_ID}_hist.nc")
STATIC   = f"{DRIVE_ROOT}/static_HR_v6.nc"

pipeline = NetCDFDataPipeline(
    lr_path=LR_PATH, hr_path=HR_PATH,
    static_path=STATIC if Path(STATIC).exists() else None,
    seq_len=int(CONFIG.data.seq_len),
    baseline_strategy=str(CONFIG.data.baseline_strategy),
    baseline_factor=int(CONFIG.data.baseline_factor),
    normalize=bool(CONFIG.data.normalize),
    nan_fill_strategy=str(CONFIG.data.nan_fill_strategy),
    precipitation_delta=float(CONFIG.data.precipitation_delta),
    lr_variables=list(CONFIG.data.lr_variables),
    hr_variables=list(CONFIG.data.hr_variables),
)
builder = HeteroGraphBuilder(CONFIG.graph)

allowed  = set(builder.dynamic_node_types) | set(builder.static_node_types)
enc_cfgs = [IntelligibleVariableConfig(name=m.name, meta_path=(m.src, m.relation, m.target),
                                       pool=m.get("pool", "mean"))
            for m in CONFIG.encoder.metapaths if m.src in allowed and m.target in allowed]
if pipeline.get_static_dataset() is not None:
    enc_cfgs.append(IntelligibleVariableConfig(name="static",
                    meta_path=("SP_HR", "causes", "GP850"), pool="mean"))

encoder = IntelligibleVariableEncoder(configs=enc_cfgs,
    hidden_dim=int(CONFIG.encoder.hidden_dim),
    conditioning_dim=int(CONFIG.encoder.conditioning_dim)).to(DEVICE)

_probe = builder.lr_grid_to_nodes(torch.zeros(len(CONFIG.data.lr_variables),
                                              *tuple(CONFIG.graph.lr_shape)))
rcn_cell = RCNCell(num_vars=len(enc_cfgs), hidden_dim=int(CONFIG.rcn.hidden_dim),
                   driver_dim=_probe.shape[-1], reconstruction_dim=_probe.shape[-1],
                   dropout=float(CONFIG.rcn.dropout)).to(DEVICE)
rcn_runner = RCNSequenceRunner(rcn_cell, detach_interval=CONFIG.rcn.get("detach_interval"))

rh = CONFIG.two_stage.regression_head
regression_head = GraphToGridDecoder(d_model=int(rh.d_model),
    hr_h=int(CONFIG.diffusion.height), hr_w=int(CONFIG.diffusion.width),
    intermediate_h=int(rh.intermediate_h), intermediate_w=int(rh.intermediate_w),
    n_heads=int(rh.n_heads), refine_channels=int(rh.refine_channels),
    output_channels=1).to(DEVICE)

skip_block = None
if HAS_SKIP:
    from st_cdgm.models import ConditionalSkipBlock
    sc = CONFIG.v5.skip_connection
    skip_block = ConditionalSkipBlock(
        lr_channels=len(CONFIG.data.lr_variables),
        hr_shape=(int(CONFIG.diffusion.height), int(CONFIG.diffusion.width)),
        alpha_floor=float(sc.alpha_floor)).to(DEVICE)

def _load(mod, key):
    if key in ck and mod is not None:
        mod.load_state_dict(ck[key]); mod.eval(); print(f"   {key} chargé")
_load(encoder, "encoder_state_dict")
_load(rcn_cell, "rcn_cell_state_dict")
_load(regression_head, "regression_head_state_dict")
_load(skip_block, "skip_block_state_dict")
print("[Cell 3] pile étage 1 reconstruite")
"""

C4 = """# >>> Cell 4 : Inférence étage 1 — collecte mu_causal, mu_direct, alpha, cible
import torch.nn.functional as F

@torch.no_grad()
def run_stage1(n_days=N_DAYS):
    \"\"\"Renvoie dict de tableaux [N,H,W] (log1p) + alpha [N].\"\"\"
    ds = pipeline.get_dataset(split="train")     # in-distribution
    out = {k: [] for k in ("mu_causal", "mu_blend", "mu_direct",
                           "baseline_log", "target_log")}
    alphas = []
    for i in range(min(n_days, len(ds))):
        s = ds[i]
        batch = pipeline.convert_sample_to_batch(s, device=DEVICE)
        H_T = rcn_runner(encoder, builder, batch)["H_T"]
        mu_c = regression_head(H_T)                              # [1,1,H,W]
        if skip_block is not None:
            lr_last = batch["lr_grid"][:, -1] if batch["lr_grid"].dim() == 5 else batch["lr_grid"]
            mu_b, a = skip_block(lr_last, mu_c)
            mu_d = (mu_b - a.view(-1, 1, 1, 1) * mu_c) / (1 - a.view(-1, 1, 1, 1) + 1e-8)
            alphas.append(float(a.mean()))
        else:
            mu_b, mu_d = mu_c, torch.zeros_like(mu_c); alphas.append(1.0)
        out["mu_causal"].append(mu_c.squeeze().cpu().numpy())
        out["mu_blend"].append(mu_b.squeeze().cpu().numpy())
        out["mu_direct"].append(mu_d.squeeze().cpu().numpy())
        out["baseline_log"].append(batch["baseline_log"].squeeze().cpu().numpy())
        out["target_log"].append(batch["target_log"].squeeze().cpu().numpy())
        if (i + 1) % 50 == 0:
            print(f"   {i+1}/{n_days}", flush=True)
    res = {k: np.stack(v) for k, v in out.items()}
    res["alpha"] = np.array(alphas)
    return res

S1 = run_stage1()
print({k: v.shape for k, v in S1.items()})
np.savez_compressed("results/stage1_diag_fields.npz", **S1)
print("[Cell 4] champs étage 1 sauvegardés")
"""

C5 = """# >>> Cell 5 : DIAGNOSTIC A — prédictibilité résiduelle (LE test décisif)
# Question : l'étage 1 détruit-il de l'information récupérable depuis y ?
#   R² hors échantillon ~ 0  => le lissage EST la vraie moyenne conditionnelle.
#                               Ne pas durcir l'étage 1, le problème est ailleurs.
#   R² nettement > 0        => l'étage 1 lisse TROP. Le décodeur est en cause.
import torch.nn as nn

to_mm = lambda z: np.expm1(np.clip(z, -20, 20))
x_mm  = to_mm(S1["baseline_log"] + S1["target_log"])
mu_mm = to_mm(S1["baseline_log"] + S1["mu_blend"])
resid = x_mm - mu_mm                                   # ce que l'étage 2 doit produire

# prédicteur simple du résidu à partir des champs LR (upsamplés)
@torch.no_grad()
def lr_stack(n):
    ds = pipeline.get_dataset(split="train"); L = []
    for i in range(n):
        b = pipeline.convert_sample_to_batch(ds[i], device="cpu")
        g = b["lr_grid"]
        g = g[:, -1] if g.dim() == 5 else g
        L.append(g.squeeze(0).numpy())
    return np.stack(L)

Y = lr_stack(len(resid))                                # [N,C,h,w]
Yt = torch.from_numpy(Y).float()
Yup = F.interpolate(Yt, size=resid.shape[-2:], mode="bilinear", align_corners=False)

ntr = int(0.7 * len(resid))
net = nn.Sequential(nn.Conv2d(Yup.shape[1], 64, 3, padding=1), nn.GELU(),
                    nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
                    nn.Conv2d(64, 1, 3, padding=1)).to(DEVICE)
opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
Rt = torch.from_numpy(resid).float().unsqueeze(1)

for ep in range(40):
    net.train(); perm = torch.randperm(ntr)
    for j in range(0, ntr, 8):
        idx = perm[j:j+8]
        p = net(Yup[idx].to(DEVICE)); loss = ((p - Rt[idx].to(DEVICE)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    if (ep + 1) % 10 == 0: print(f"   ep{ep+1} loss={loss.item():.4f}", flush=True)

net.eval()
with torch.no_grad():
    pred = net(Yup[ntr:].to(DEVICE)).cpu().numpy().squeeze(1)
rte = resid[ntr:]
r2 = 1 - ((rte - pred) ** 2).sum() / max(((rte - rte.mean()) ** 2).sum(), 1e-9)
print(f"\\n=== DIAGNOSTIC A ===\\n  R² hors échantillon de g(y) -> résidu : {r2:.4f}")
print("  R² ~ 0      => lissage = vraie moyenne conditionnelle, NE PAS durcir l'étage 1")
print("  R² > ~0.05  => l'étage 1 lisse TROP (information récupérable perdue)")
DIAG = {"A_residual_R2": float(r2)}
"""

C6 = """# >>> Cell 6 : DIAGNOSTIC B — alpha conditionné à l'intensité (bimodalité ?)
import matplotlib.pyplot as plt

if skip_block is None:
    print("pas de skip_block dans ce checkpoint — diagnostic B sans objet")
else:
    a = S1["alpha"]
    inten = np.array([np.quantile(f, 0.99) for f in x_mm])   # p99 du jour
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(a, bins=40); ax[0].axvline(float(CONFIG.v5.skip_connection.alpha_floor),
               color="r", ls="--", label="alpha_floor")
    ax[0].set_xlabel("alpha"); ax[0].set_ylabel("jours"); ax[0].legend()
    ax[0].set_title("Distribution de alpha")
    ax[1].scatter(inten, a, s=6, alpha=.5)
    ax[1].set_xlabel("p99 de la précipitation du jour (mm/j)"); ax[1].set_ylabel("alpha")
    ax[1].set_title("alpha vs intensité de l'événement")
    plt.tight_layout(); plt.savefig("results/diag_B_alpha.png", dpi=130); plt.show()

    hi = inten > np.quantile(inten, 0.9)
    print(f"\\n=== DIAGNOSTIC B ===")
    print(f"  alpha moyen (tous jours)      : {a.mean():.3f}")
    print(f"  alpha moyen (10% plus intenses): {a[hi].mean():.3f}")
    print(f"  alpha min                      : {a.min():.3f}")
    print("  => si alpha chute sur les jours intenses, la contrainte de MOYENNE")
    print("     de batch est satisfaite tout en amputant les extrêmes.")
    DIAG.update({"B_alpha_mean": float(a.mean()),
                 "B_alpha_mean_top10pct_intensity": float(a[hi].mean()),
                 "B_alpha_min": float(a.min())})
"""

C7 = """# >>> Cell 7 : DIAGNOSTIC C — RAPSD : le skip block ou le décodeur ?
def rapsd(f):
    f = f - f.mean(axis=(-2, -1), keepdims=True)
    P = np.abs(np.fft.rfft2(f)) ** 2
    H, W = f.shape[-2], f.shape[-1]
    ky = np.fft.fftfreq(H)[:, None]; kx = np.fft.rfftfreq(W)[None, :]
    kr = np.sqrt(ky ** 2 + kx ** 2); nb = 40
    kb = np.clip((kr / kr.max() * nb).astype(int), 0, nb - 1)
    return np.array([P[..., kb == i].mean() if (kb == i).any() else np.nan
                     for i in range(nb)])

Sx  = rapsd(S1["target_log"]); Sc = rapsd(S1["mu_causal"])
Sb  = rapsd(S1["mu_blend"])
ratio_c, ratio_b = Sc / Sx, Sb / Sx
a_mean = float(S1["alpha"].mean())

plt.figure(figsize=(7, 4.5))
plt.semilogy(ratio_c, label="mu_causal / cible")
plt.semilogy(ratio_b, label="mu_blend / cible")
plt.axhline(a_mean ** 2, color="r", ls="--", label=f"alpha² = {a_mean**2:.3f}")
plt.xlabel("bin de nombre d'onde (grandes -> petites échelles)")
plt.ylabel("ratio de puissance"); plt.legend(); plt.grid(alpha=.3)
plt.title("Signature spectrale : plat ≈ alpha² (skip) vs décroissant (décodeur)")
plt.tight_layout(); plt.savefig("results/diag_C_rapsd.png", dpi=130); plt.show()

hf = slice(len(ratio_b) // 2, None)
flat = float(np.nanstd(ratio_b[hf]) / max(np.nanmean(ratio_b[hf]), 1e-9))
print(f"\\n=== DIAGNOSTIC C ===")
print(f"  ratio HF moyen (mu_blend/cible) : {np.nanmean(ratio_b[hf]):.4f}")
print(f"  alpha² attendu si skip coupable  : {a_mean**2:.4f}")
print(f"  variabilité relative du ratio HF : {flat:.3f}  (<0.2 => PLAT => skip)")
DIAG.update({"C_hf_ratio": float(np.nanmean(ratio_b[hf])),
             "C_alpha2": a_mean ** 2, "C_hf_flatness": flat})
"""

C8 = """# >>> Cell 8 : DIAGNOSTIC D — F1@p99 par membre vs sur la moyenne d'ensemble
# Le code d'éval actuel calcule sur members_mm.mean(dim=0) : un champ lissé
# une SECONDE fois. Cette cellule mesure le coût de ce choix.
clim99 = np.quantile(x_mm, 0.99, axis=0)
truth  = x_mm > clim99[None]

def f1(pred):
    pe = pred > clim99[None]
    tp = float((pe & truth).sum()); fp = float((pe & ~truth).sum()); fn = float((~pe & truth).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1e-9)

# NOTE : nécessite l'étage 2 pour un vrai ensemble. À défaut, on borne l'effet
# en comparant mu (déterministe) et une perturbation stochastique calibrée.
sig = float(np.std(x_mm - mu_mm))
rng = np.random.default_rng(SEED)
members = np.stack([mu_mm + rng.normal(0, sig, mu_mm.shape) for _ in range(K_MEMBERS)])
f1_mean   = f1(members.mean(axis=0))
f1_single = float(np.mean([f1(m) for m in members]))
f1_q90    = f1(np.quantile(members, 0.90, axis=0))
print(f"\\n=== DIAGNOSTIC D ===")
print(f"  F1@p99 sur la MOYENNE d'ensemble : {f1_mean:.4f}   <- convention actuelle")
print(f"  F1@p99 par membre (moyenne)      : {f1_single:.4f}")
print(f"  F1@p99 sur le quantile 90 %      : {f1_q90:.4f}")
print("  => si l'écart est net, la métrique pénalise la moyenne, pas le modèle.")
DIAG.update({"D_f1_ensemble_mean": f1_mean, "D_f1_per_member": f1_single,
             "D_f1_q90": f1_q90})
"""

C9 = """# >>> Cell 9 : DIAGNOSTIC E — biais de Jensen du modèle réel
# Confirme sur le modèle entraîné les 10-18 % mesurés sur données seules.
qs = np.quantile(mu_mm, np.linspace(0, 1, 13)); qs = np.unique(qs)
bid = np.clip(np.digitize(mu_mm.ravel(), qs[1:-1]), 0, len(qs) - 2)
xr_, mr_ = x_mm.ravel(), mu_mm.ravel()
rows = []
for k in range(len(qs) - 1):
    s = bid == k
    if s.sum() < 1000: continue
    rows.append(dict(bin=k, mu_lo=float(qs[k]), mu_hi=float(qs[k+1]),
                     n=int(s.sum()), obs_mean=float(xr_[s].mean()),
                     pred_mean=float(mr_[s].mean()),
                     bias_pct=100*(float(xr_[s].mean())-float(mr_[s].mean()))
                              / max(float(xr_[s].mean()), 1e-9)))
print("\\n=== DIAGNOSTIC E — biais conditionnel du modèle ===")
print(f"{'bin mu (mm/j)':>20s} {'obs':>9s} {'pred':>9s} {'biais':>8s}")
for r in rows:
    print(f"{r['mu_lo']:8.2f}-{r['mu_hi']:8.2f} {r['obs_mean']:9.3f} "
          f"{r['pred_mean']:9.3f} {r['bias_pct']:7.1f}%")
DIAG["E_bias_bins"] = rows

json.dump(DIAG, open("results/diag_stage1_model.json", "w"), indent=2)
print("\\n=== ECRIT results/diag_stage1_model.json ===")
"""

C10 = """# >>> Cell 10 : SYNTHÈSE — arbre de décision
print("=" * 68)
print("SYNTHÈSE DES DIAGNOSTICS ÉTAGE 1")
print("=" * 68)
r2 = DIAG.get("A_residual_R2", float("nan"))
print(f"\\nA. Prédictibilité résiduelle  R² = {r2:.4f}")
if r2 < 0.02:
    print("   -> le lissage EST la vraie moyenne conditionnelle.")
    print("      NE PAS durcir l'étage 1. Travailler l'étage 2 (sous-dispersion).")
else:
    print("   -> l'étage 1 lisse TROP : information récupérable perdue.")
    print("      PRIORITÉ : résolution du décodeur (43x45 -> 86x90) + requêtes")
    print("      dépendantes de l'entrée + encodage positionnel.")

if "C_hf_flatness" in DIAG:
    print(f"\\nC. Signature spectrale  platitude = {DIAG['C_hf_flatness']:.3f}, "
          f"ratio HF = {DIAG['C_hf_ratio']:.4f}, alpha² = {DIAG['C_alpha2']:.4f}")
    if DIAG["C_hf_flatness"] < 0.2 and abs(DIAG["C_hf_ratio"] - DIAG["C_alpha2"]) < 0.1:
        print("   -> déficit PLAT ≈ alpha² : le skip block est la cause.")
    else:
        print("   -> déficit CROISSANT avec k : le DÉCODEUR est la cause.")
        print("      (cohérent avec le §9.3 : prédictibilité à 9-13 km, décodeur à 48 km)")

if "B_alpha_mean" in DIAG:
    d = DIAG["B_alpha_mean"] - DIAG["B_alpha_mean_top10pct_intensity"]
    print(f"\\nB. alpha : moyen {DIAG['B_alpha_mean']:.3f}, "
          f"jours intenses {DIAG['B_alpha_mean_top10pct_intensity']:.3f} (écart {d:+.3f})")
    if d > 0.1:
        print("   -> BIMODALITÉ confirmée : la contrainte de moyenne de batch est")
        print("      satisfaite alors qu'alpha chute sur les jours extrêmes.")

print(f"\\nD. F1@p99 moyenne d'ensemble {DIAG.get('D_f1_ensemble_mean', float('nan')):.4f} "
      f"vs par membre {DIAG.get('D_f1_per_member', float('nan')):.4f}")
print("\\nActions de référence : §9.6 de docs/architecture_v8_design.md")
print("=" * 68)
"""

cells = [md(C0), code(C1), code(C2), code(C3), code(C4), code(C5),
         code(C6), code(C7), code(C8), code(C9), code(C10)]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"},
                   "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"écrit : {OUT}  ({len(cells)} cellules)")
