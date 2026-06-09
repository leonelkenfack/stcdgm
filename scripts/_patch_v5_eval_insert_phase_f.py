"""Patch Phase F : insere une cellule de fine-tune dans le notebook d'eval.

Inseres apres Cell 4 (chargement des stacks) :
- Cell MD : explication de Phase F
- Cell code : fine-tune avec switch SMOKE_TEST (2 epochs) vs full (25 epochs)

Usage :
    python scripts/_patch_v5_eval_insert_phase_f.py
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

BACKUP = Path("st_cdgm_v5_evaluation.ipynb.bak_phaseF_insert")
if not BACKUP.exists():
    BACKUP.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] Backup cree : {BACKUP}")

# Sanity : on n'insere qu'une fois
if any("Phase F : fine-tune Bundle B + CASTLE + G_phys" in "".join(c["source"])
       for c in nb["cells"]):
    print("[SKIP] Cellule Phase F deja presente, abort")
    sys.exit(0)

# === Cell MD : explication ===
CELL_MD = """---

## Phase F (post-V5-mini, 2026-06-10) — Fine-tune Bundle B + CASTLE + G_phys

**Cellule optionnelle** qui re-entraine `stack_v5` (encoder + RCN + reg_head + skip si dispo)
avec toutes les modifs Phases A-E :

- **CASTLE-style joint prediction anchoring** sur A_dag (Phase A)
- **Weighted MSE power-law** + **L1 cosine annealing** (Phase B)
- **Pinball quantile loss** (τ=0.95, 0.99) + **CC regularizer** + **spectral high-k** (Phase C)
- **Masque physique G_phys** (Phase D) — couplage QG descendant
- **TailStratifiedSampler** (Phase E) — 30 % extrêmes garantis par batch

Stage 2 (diffusion) est **figé**. `sigma_data` est **recalibré** en fin de run.

Pour éviter de claquer 4h en aveugle, le code expose un flag `SMOKE_TEST` :
- `SMOKE_TEST = True`  → 2 epochs (~20-30 min sur GPU T4, valide la pipeline)
- `SMOKE_TEST = False` → 25 epochs (~30-50 min GPU T4, ~3-4h CPU) — le vrai run

**Une fois le fine-tune termine**, change `CHECKPOINT_NAME = "epoch_finetuned"` en Cell 2 et
re-execute les cellules a partir de Cell 4 pour evaluer le nouveau modele.

Voir `architecture_journey.md` §12 pour le plan complet et les cibles d'amelioration.
"""

# === Cell code : fine-tune ===
CELL_CODE = '''# === Phase F : fine-tune Bundle B + CASTLE + G_phys ===
# Met SMOKE_TEST = True pour un test rapide 2 epochs avant le vrai run.
SMOKE_TEST = True   # <<< change a False pour le vrai run 25 epochs

EPOCHS = 2 if SMOKE_TEST else 25
SANITY_EVERY = 1 if SMOKE_TEST else 5

from scripts.finetune_stage1_bundle_b import finetune_bundle_b

# 1. Construire un train_dataset distinct du test_dataset (stride=1)
print(f"[Phase F] Construction du train_dataset (ACCESS-CM2)...")
pipe_train = make_pipeline(GCM_REGISTRY["ACCESS-CM2"][0], GCM_REGISTRY["ACCESS-CM2"][1])
train_dataset = pipe_train.build_sequence_dataset(
    seq_len=int(CONFIG.data.seq_len),
    stride=1,
    as_torch=True,
)
print(f"  train_dataset : {len(train_dataset)} samples")
print(f"  val_dataset (=test_dataset) : {len(test_dataset)} samples")
print(f"  SMOKE_TEST = {SMOKE_TEST}  ->  EPOCHS = {EPOCHS}")
print()

# 2. Lancer le fine-tune
result = finetune_bundle_b(
    stack=stack_v5,
    builder=builder,
    train_dataset=train_dataset,
    val_dataset=test_dataset,
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
print("  3. Re-execute Cell 4 (recharge stack), puis Cells 6, 8, 10, 12, 14, 16, 18 (Phases 6-12)")
print("  4. Compare via : !python -m scripts.compare_eval_results ...")
'''


def _make_cell(cell_type, source_str):
    lines = source_str.split("\n")
    if len(lines) > 1:
        source_list = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    else:
        source_list = [source_str]
    cell = {"cell_type": cell_type, "metadata": {}, "source": source_list}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


# Trouve Cell 4 (la code cell qui contient build_stack)
insert_after = None
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if c["cell_type"] == "code" and "stack_v5 = build_stack" in src:
        insert_after = i
        break

if insert_after is None:
    print("[KO] Impossible de trouver la cell de build_stack (Cell 4)")
    sys.exit(1)

print(f"[OK] Cell {insert_after} identifiee comme celle de chargement des stacks")
print(f"     Insertion de 2 cellules apres (decalage des cells suivantes de +2)")

new_cells = [
    _make_cell("markdown", CELL_MD),
    _make_cell("code", CELL_CODE),
]
nb["cells"] = nb["cells"][:insert_after + 1] + new_cells + nb["cells"][insert_after + 1:]

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[OK] 2 cellules inserees")
print(f"     Notebook : {len(nb['cells'])} cellules au total")
