# >>> Cell 2 : configuration V8 + interrupteurs + garde holdout
from omegaconf import OmegaConf
from st_cdgm.data.derived import FREE_NODES

CONFIG = OmegaConf.load("config/training_config.yaml")

# --- Etage 2 : la MEME architecture que les modeles auxquels on se compare.
# Les metriques de reference (ORACLE, CorrDiff, V6') ont toutes ete produites
# avec l'UNet CorrDiff-Normal : 4 niveaux, [128,256,256,256], ~50 M parametres
# (champ `config_block_out_channels` de leurs JSON de metriques). Le bloc
# `diffusion` du YAML de base decrit un UNet MINIMAL de 1 M parametres : le
# garder ferait mesurer la taille de l'UNet, pas l'apport de V8. Le merge
# apporte aussi S_churn=40 et la tail_weight 8/25 du protocole de reference.
_S2_REF = OmegaConf.load("config/training_config_corrdiff_normal.yaml")
CONFIG.diffusion = OmegaConf.merge(CONFIG.diffusion, _S2_REF.diffusion)
print(f"etage 2 : UNet {list(CONFIG.diffusion.unet_kwargs.block_out_channels)} "
      f"(CorrDiff-Normal, celui des references)")

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
# Etage 2 : le budget se compte en TIRAGES (echantillons vus), la seule unite
# invariante. V5 : 250 epoques x 5 467 fenetres (stride 2) = ~1,37 M tirages.
# Ici le cache tient ~2 734 fenetres (stride 4), donc 500 passes = le meme
# budget. La valeur initiale, 20, venait du cap `stage2.epochs_max` du YAML de
# base et ne donnait que ~55 000 tirages : vingt-cinq fois moins, sur un
# modele de diffusion. Le nombre de PAS depend en plus du batch (Cell 9) ; il
# est affiche a la premiere epoque.
EPOCHS_S2   = int(os.environ.get("V8_EPOCHS_S2", 500))
N_EVAL      = int(os.environ.get("V8_N_EVAL", 300))   # audit Jensen (Cell 7)

# --- Protocole d'evaluation. Ces valeurs ne sont PAS libres : ce sont celles
#     sous lesquelles V6', ORACLE (V5) et CorrDiff ont ete mesures dans le
#     3-way. Les changer rend la Cell 11 incomparable — elle le detecte et
#     refuse le verdict plutot que d'aligner des nombres de protocoles
#     differents. cfg 0.0 = conditioned-only : identique a 1.0 sur edm_karras,
#     sans le double forward CFG.
K_VERDICT   = int(os.environ.get("V8_K", 32))
NUM_STEPS   = int(os.environ.get("V8_NUM_STEPS", 24))
CFG_SCALE   = 0.0
EVAL_BATCH  = int(os.environ.get("V8_EVAL_BATCH", 16))
SEQ_LEN     = int(CONFIG.data.seq_len)
HIDDEN      = int(CONFIG.rcn.hidden_dim)
LR_SHAPE    = (23, 26)
HR_SHAPE    = (172, 179)
# Checkpoints sur DRIVE en Colab. `checkpoints_v8/` sous /content est efface
# avec la VM : sur un run de plusieurs heures, une deconnexion perdrait
# tout. Le clone du code reste sur le SSD local (xarray y est ~20x plus
# rapide), mais les poids doivent survivre a la session.
CKPT_DIR = ((Path(DRIVE_ROOT) / "checkpoints_v8") if IN_COLAB
            else Path("checkpoints_v8"))
CKPT_DIR.mkdir(parents=True, exist_ok=True)
print(f"checkpoints : {CKPT_DIR}")
RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)

# --- Garde holdout. NorESM2-MM est pre-enregistre comme intouchable : une
#     seule evaluation finale, jamais pendant le developpement. Toute cellule
#     qui ouvre un fichier passe par guard().
HOLDOUT = "NorESM2"


def guard(p):
    assert HOLDOUT not in str(p), f"STOP - {p} touche le holdout {HOLDOUT}-MM."
    return Path(p)


print("13 noeuds :", list(FREE_NODES))
print("Les chemins de donnees sont resolus en Cell 2b (telechargement si absent).")
