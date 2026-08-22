# >>> Cell 2 : configuration V8 + interrupteurs + garde holdout
from omegaconf import OmegaConf
from st_cdgm.data.derived import FREE_NODES

CONFIG = OmegaConf.load("config/training_config.yaml")

# --- Interrupteurs. Tout a False donne approximativement la pile V5.
#     P1 exige UN SEUL changement par run pour pouvoir attribuer l'effet.
V8 = OmegaConf.create(dict(
    free_nodes      = True,   # 13 noeuds libres derives au lieu des 15 bruts
    diagonal_driver = True,   # V7-M2 : chaque variable ne voit que son canal
    instantaneous   = True,   # C2/C5 : A(0) contemporaine en plus de A(tau>=1)
    edge_prior      = True,   # C7 : prior 3 niveaux, annele par niveau
    spatial_queries = True,   # A2a : requetes du decodeur dependantes de l'entree
    bernoulli_gamma = True,   # A3 : vraisemblance BG, ancre mu = p*alpha*beta
    jensen          = True,   # A1 : correction du retour log1p -> mm
))
print(OmegaConf.to_yaml(V8))

# --- Budget. Pour un smoke, reduire EPOCHS ; ne JAMAIS toucher aux seuils.
EPOCHS_S1   = int(os.environ.get("V8_EPOCHS_S1", 30))
EPOCHS_S2   = int(os.environ.get("V8_EPOCHS_S2", 20))
N_EVAL      = int(os.environ.get("V8_N_EVAL", 300))
K_ENSEMBLE  = int(os.environ.get("V8_ENSEMBLE", 16))
SEQ_LEN     = int(CONFIG.data.seq_len)
HIDDEN      = int(CONFIG.rcn.hidden_dim)
LR_SHAPE    = (23, 26)
HR_SHAPE    = (172, 179)
CKPT_DIR    = Path("checkpoints_v8"); CKPT_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)

# --- Donnees : EXACTEMENT celles utilisees jusqu'ici, rien de plus.
DATA        = Path("data/raw")
LR_PATH     = DATA / "train" / "predictor_ACCESS-CM2_hist.nc"
HR_PATH     = DATA / "train" / "pr_ACCESS-CM2_hist.nc"
STATIC_PATH = DATA / "static_predictors" / "ERA5_eval_ccam_12km.198110_NZ_Invariant.nc"

# --- Garde holdout. NorESM2-MM est pre-enregistre comme intouchable : une
#     seule evaluation finale, jamais pendant le developpement.
HOLDOUT = "NorESM2"

def guard(p):
    assert HOLDOUT not in str(p), f"STOP - {p} touche le holdout {HOLDOUT}-MM."
    return Path(p)

for _p in (LR_PATH, HR_PATH, STATIC_PATH):
    guard(_p)
    assert _p.exists(), f"{_p} absent - verifier data/raw/"

print("donnees   :", LR_PATH.name, "|", HR_PATH.name, "|", STATIC_PATH.name)
print("13 noeuds :", list(FREE_NODES))
