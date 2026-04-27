"""
Cell 34 (build models) filtre silencieusement les metapaths dont les
nœuds (src/target) ne sont pas dans ``builder.dynamic_node_types |
builder.static_node_types``. Si tous les metapaths sont filtrés (ex.
``include_mid_layer=False``, typo dans le YAML, casse différente), on a :
  - encoder_configs vide → IntelligibleVariableEncoder lève ValueError
    "Au moins une configuration de variable intelligible est requise."
  - OU un sous-ensemble plus petit qu'attendu, num_vars réduit silencieusement
    → projection_class_embeddings_input_dim wrong → UNet wrong → downstream
    crashes plus tard sans message clair sur la cause racine.

Patch : remplacer la list-comprehension silencieuse par une boucle
verbose qui :
  1. logge ``allowed_nodes`` du builder
  2. pour CHAQUE metapath : kept ou dropped (avec raison)
  3. raise loudly si le résultat est vide ou si moins que ``len(YAML)``
     metapaths survivent quand ``allow_missing_metapaths: false``

Idempotent par sentinelle ``# >>> CELL34_VERBOSE_FILTER``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "st_cdgm_training_evaluation.ipynb"
SENTINEL = "# >>> CELL34_VERBOSE_FILTER"

OLD_BLOCK = """allowed_nodes = set(builder.dynamic_node_types) | set(builder.static_node_types)
encoder_configs = [
    IntelligibleVariableConfig(
        name=mp.name,
        meta_path=(mp.src, mp.relation, mp.target),
        pool=mp.get("pool", "mean"),
    )
    for mp in CONFIG.encoder.metapaths
    if mp.src in allowed_nodes and mp.target in allowed_nodes
]"""

NEW_BLOCK = """# >>> CELL34_VERBOSE_FILTER
# Construction verbose des configs metapaths : log de chaque kept/dropped
# pour éviter le silent-filter quand un nœud du YAML n'est pas dans le
# graphe (typo, include_mid_layer=False, etc.).
allowed_nodes = set(builder.dynamic_node_types) | set(builder.static_node_types)
print(f"🔎 Nodes disponibles dans le graphe: dyn={builder.dynamic_node_types} static={builder.static_node_types}")

encoder_configs = []
_kept, _dropped = [], []
for _mp in CONFIG.encoder.metapaths:
    _src, _rel, _tgt = _mp.src, _mp.relation, _mp.target
    if _src in allowed_nodes and _tgt in allowed_nodes:
        encoder_configs.append(IntelligibleVariableConfig(
            name=_mp.name,
            meta_path=(_src, _rel, _tgt),
            pool=_mp.get("pool", "mean"),
        ))
        _kept.append(_mp.name)
    else:
        _missing = [n for n in (_src, _tgt) if n not in allowed_nodes]
        _dropped.append(f"{_mp.name} ({_src}→{_tgt}) — manque: {_missing}")

print(f"🔎 Metapaths YAML→retenus : {len(_kept)}/{len(CONFIG.encoder.metapaths)}")
for _name in _kept:
    print(f"     ✓ {_name}")
for _msg in _dropped:
    print(f"     ✗ {_msg}")

# Fail loudly si on a perdu des metapaths sans le permettre explicitement.
_allow_missing = bool(CONFIG.encoder.get("allow_missing_metapaths", False))
if not _allow_missing and _dropped:
    raise RuntimeError(
        f"{len(_dropped)} metapath(s) du YAML ne match pas le graphe "
        f"(allow_missing_metapaths=false). Soit corriger les noms (src/target) "
        f"soit activer allow_missing_metapaths=true. "
        f"Dropped: {_dropped}"
    )
if not encoder_configs:
    raise RuntimeError(
        "Aucun metapath n'a survécu au filtre — l'encoder ne peut pas être créé. "
        f"Vérifier que CONFIG.graph.include_mid_layer={CONFIG.graph.include_mid_layer} "
        f"et que les noms (src/target) des metapaths YAML matchent: {sorted(allowed_nodes)}"
    )"""


def main() -> int:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    target = None
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if "Intelligible Variable Encoder" in src and "encoder_configs" in src:
            target = i
            break

    if target is None:
        print("[ERROR] Cell construction Encoder/RCN/Diffusion introuvable.")
        return 2

    src = "".join(cells[target].get("source", []))

    if SENTINEL in src:
        print(f"✓ Cell {target} déjà patchée (verbose filter).")
        return 0

    if OLD_BLOCK not in src:
        print(f"[ERROR] Bloc original (allowed_nodes + list comprehension) non trouvé.")
        print("        Le cell a peut-être été manuellement modifiée — patch à adapter.")
        return 3

    new_src = src.replace(OLD_BLOCK, NEW_BLOCK, 1)
    cells[target]["source"] = new_src.splitlines(keepends=True)
    cells[target]["outputs"] = []
    cells[target]["execution_count"] = None

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  ↪ Cell {target} : verbose metapath filter + fail-loudly")
    print(f"💾 Saved {NB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
