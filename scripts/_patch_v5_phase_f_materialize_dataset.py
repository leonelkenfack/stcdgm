"""Patch Phase F cell : materialize l'IterableDataset en map-style dataset.

build_sequence_dataset retourne un ResDiffIterableDataset qui n'a pas de
__len__ ni __getitem__, ce qui empeche TailStratifiedSampler de fonctionner
et compute_sample_max_values d'iterer par index.

Fix : on materialize l'iterable en liste (365 samples ~ 1 GB en RAM, le
user a 500 GB) et on l'enveloppe dans un Dataset map-style simple.
"""
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NB = Path("st_cdgm_v5_evaluation.ipynb")
with NB.open(encoding="utf-8") as f:
    nb = json.load(f)

BACKUP = Path("st_cdgm_v5_evaluation.ipynb.bak_phaseF_materialize")
if not BACKUP.exists():
    BACKUP.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] Backup cree : {BACKUP}")

# Trouve la cellule Phase F
phase_f_idx = None
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if c["cell_type"] == "code" and "Phase F : fine-tune Bundle B + CASTLE + G_phys" in src:
        phase_f_idx = i
        break

if phase_f_idx is None:
    print("[KO] Cellule Phase F introuvable")
    sys.exit(1)

print(f"[OK] Cellule Phase F a l'index {phase_f_idx}")

NEW_CELL = '''# === Phase F : fine-tune Bundle B + CASTLE + G_phys ===
# Met SMOKE_TEST = True pour un test rapide 2 epochs avant le vrai run.
SMOKE_TEST = True   # <<< change a False pour le vrai run 25 epochs

EPOCHS = 2 if SMOKE_TEST else 25
SANITY_EVERY = 1 if SMOKE_TEST else 5

from scripts.finetune_stage1_bundle_b import finetune_bundle_b
from torch.utils.data import Dataset as _TorchDataset


class _MapStyleListDataset(_TorchDataset):
    """Wrapper map-style autour d'une liste de samples materialises.

    TailStratifiedSampler et compute_sample_max_values necessitent __len__ et
    __getitem__, ce que ResDiffIterableDataset ne fournit pas.
    """
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        return self.samples[i]


# 1. Construire un train_dataset distinct du test_dataset (stride=1)
print(f"[Phase F] Construction du train_dataset (ACCESS-CM2)...")
pipe_train = make_pipeline(GCM_REGISTRY["ACCESS-CM2"][0], GCM_REGISTRY["ACCESS-CM2"][1])
train_iter = pipe_train.build_sequence_dataset(
    seq_len=int(CONFIG.data.seq_len),
    stride=1,
    as_torch=True,
)

# 2. Materialize l'IterableDataset en liste pour usage map-style
print(f"  Materialization de l'iterable en liste (peut prendre ~1-2 min)...")
import time as _t
_t0 = _t.time()
_samples = list(train_iter)
print(f"  [OK] {len(_samples)} samples materialises en {_t.time()-_t0:.1f}s")
train_dataset = _MapStyleListDataset(_samples)

# Idem pour test_dataset s'il est iterable
if not hasattr(test_dataset, "__len__"):
    print(f"  Materialization du val_dataset (test_dataset) aussi...")
    _val_samples = list(test_dataset)
    val_dataset = _MapStyleListDataset(_val_samples)
    print(f"  [OK] {len(_val_samples)} val samples materialises")
else:
    val_dataset = test_dataset

print(f"  train_dataset : {len(train_dataset)} samples")
print(f"  val_dataset   : {len(val_dataset)} samples")
print(f"  SMOKE_TEST = {SMOKE_TEST}  ->  EPOCHS = {EPOCHS}")
print()

# 3. Lancer le fine-tune
result = finetune_bundle_b(
    stack=stack_v5,
    builder=builder,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    CONFIG=CONFIG,
    DEVICE=DEVICE,
    epochs=EPOCHS,
    batch_size=8,
    ckpt_save_dir=V5_DIR,
    convert_sample_to_batch_fn=convert_sample_to_batch,
    sanity_eval_every=SANITY_EVERY,
    seed=42,
    skip_sigma_data_recalib=SMOKE_TEST,   # skip recalib en smoke test
)

print()
print("=" * 60)
print(f"[OK] Phase F terminee  ({EPOCHS} epochs)")
print(f"  Checkpoint sauvegarde : {result['final_ckpt']}")
print(f"  Nouveau sigma_data    : {result['sigma_data_new']}")
print("=" * 60)
print()
print("PROCHAINE ETAPE :")
print("  1. Si SMOKE_TEST = True : passe a SMOKE_TEST = False et relance cette cellule")
print("  2. Une fois fine-tune complet : change CHECKPOINT_NAME = \\"epoch_finetuned\\" en Cell 2")
print("  3. Re-execute Cell 4 (recharge stack), puis Cells 8, 10, 12, 14, 16, 18, 20, 22 (Phases 6-12)")
print("  4. Compare via : !python -m scripts.compare_eval_results ...")'''


nb["cells"][phase_f_idx]["source"] = [l + "\n" for l in NEW_CELL.split("\n")[:-1]] + [NEW_CELL.split("\n")[-1]]

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("[OK] Cellule Phase F mise a jour avec materialization IterableDataset -> map-style")
