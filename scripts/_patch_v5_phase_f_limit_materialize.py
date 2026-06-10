"""Patch Phase F : limite la materialization a 1500 samples pour eviter OOM.

build_sequence_dataset(stride=1) sur ACCESS-CM2 historical (~20-30 ans
journaliers) genere ~10000 sequences chevauchantes. Materializer en liste
consomme ~100 GB (10 MB/sample * 10000), depassant les 70 GB de Colab.

Fix : itertools.islice limite a MAX_TRAIN_SAMPLES = 1500 (~4 ans, ~15 GB)
ce qui reste largement suffisant pour 25 epochs de fine-tune.
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

BACKUP = Path("st_cdgm_v5_evaluation.ipynb.bak_phaseF_islice")
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

src = "".join(nb["cells"][phase_f_idx]["source"])

# Remplace le bloc materialize
OLD = '''# 2. Materialize l'IterableDataset en liste pour usage map-style
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
    val_dataset = test_dataset'''

NEW = '''# 2. Materialize l'IterableDataset en liste pour usage map-style.
#    ATTENTION : build_sequence_dataset(stride=1) sur 20-30 ans d'ACCESS-CM2
#    genere ~10000 sequences (~100 GB RAM). On limite a MAX_TRAIN_SAMPLES
#    qui represente ~4 ans avec stride=1, largement suffisant pour fine-tune.
MAX_TRAIN_SAMPLES = 750    # ~7.5 GB RAM (~2 ans stride=1), reduire a 500 si OOM
MAX_VAL_SAMPLES = 150      # ~1.5 GB RAM, suffisant pour recalibration sigma_data

print(f"  Materialization train_dataset (limite a {MAX_TRAIN_SAMPLES} samples, ~1-2 min)...")
import time as _t
import itertools as _it
_t0 = _t.time()
_samples = list(_it.islice(train_iter, MAX_TRAIN_SAMPLES))
print(f"  [OK] {len(_samples)} samples materialises en {_t.time()-_t0:.1f}s")
train_dataset = _MapStyleListDataset(_samples)

# Idem pour test_dataset s'il est iterable
if not hasattr(test_dataset, "__len__"):
    print(f"  Materialization val_dataset (limite a {MAX_VAL_SAMPLES} samples)...")
    _val_samples = list(_it.islice(test_dataset, MAX_VAL_SAMPLES))
    val_dataset = _MapStyleListDataset(_val_samples)
    print(f"  [OK] {len(_val_samples)} val samples materialises")
else:
    val_dataset = test_dataset'''

if OLD not in src:
    print("[KO] motif introuvable")
    sys.exit(1)

src_new = src.replace(OLD, NEW)
nb["cells"][phase_f_idx]["source"] = [l + "\n" for l in src_new.split("\n")[:-1]] + [src_new.split("\n")[-1]]

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("[OK] Cellule Phase F : MAX_TRAIN_SAMPLES=1500 + itertools.islice")
print("     RAM estimee : ~15 GB train + ~3 GB val = ~18 GB peak (vs 100 GB avant)")
