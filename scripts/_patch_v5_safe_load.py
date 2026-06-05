"""Fix _load dans build_stack pour gerer None / dict-like / prefixes torch.compile/DDP."""
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

BACKUP = Path("st_cdgm_v5_evaluation.ipynb.bak9")
if not BACKUP.exists():
    BACKUP.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] Backup #9 cree : {BACKUP}")

src = "".join(nb["cells"][4]["source"])

OLD_LOADER = '''    for n, m in [("encoder", enc), ("rcn_cell", rcn_cell),
                  ("regression_head", rh), ("diffusion", diff)]:
        k = f"{n}_state_dict"
        if k in ckpt:
            m.load_state_dict(ckpt[k])'''

NEW_LOADER = '''    def _safe_load(name, module):
        """Load state_dict avec checks None + dict-like + strip prefixes."""
        key = f"{name}_state_dict"
        if key not in ckpt:
            print(f"  [{name}] [WARN] {key} absent du checkpoint")
            return False
        sd = ckpt[key]
        if sd is None:
            print(f"  [{name}] [WARN] {key} is None - skip")
            return False
        if not hasattr(sd, "items"):
            print(f"  [{name}] [WARN] {key} not dict-like ({type(sd).__name__}) - skip")
            return False
        # Strip prefixes torch.compile ('_orig_mod.') et DDP ('module.')
        prefixes = ["_orig_mod.", "module."]
        stripped = {}
        for kk, vv in sd.items():
            new_k = kk
            for p in prefixes:
                if new_k.startswith(p):
                    new_k = new_k[len(p):]
            stripped[new_k] = vv
        try:
            missing, unexpected = module.load_state_dict(stripped, strict=False)
            if missing:
                print(f"  [{name}] [INFO] {len(missing)} keys manquantes (premieres : {missing[:3]})")
            if unexpected:
                print(f"  [{name}] [INFO] {len(unexpected)} keys inattendues (premieres : {unexpected[:3]})")
            return True
        except Exception as e:
            print(f"  [{name}] [ERREUR] load_state_dict: {type(e).__name__}: {e}")
            return False

    for n, m in [("encoder", enc), ("rcn_cell", rcn_cell),
                  ("regression_head", rh), ("diffusion", diff)]:
        _safe_load(n, m)'''

if OLD_LOADER not in src:
    print("[ERREUR] Bloc loader ancien introuvable.")
    sys.exit(1)

src_new = src.replace(OLD_LOADER, NEW_LOADER)

# Aussi rendre skip_block load safe
OLD_SKIP_LOAD = '''    if SKIP_AVAILABLE and "skip_block_state_dict" in ckpt:
        skip = ConditionalSkipBlock(
            lr_channels=len(CONFIG.data.lr_variables),
            hr_shape=tuple(CONFIG.graph.hr_shape),
        ).to(DEVICE)
        skip.load_state_dict(ckpt["skip_block_state_dict"])
        print(f"  [{name}] [+] skip_block ({skip.num_params()} params)")'''

NEW_SKIP_LOAD = '''    if (SKIP_AVAILABLE and "skip_block_state_dict" in ckpt
            and ckpt["skip_block_state_dict"] is not None):
        skip = ConditionalSkipBlock(
            lr_channels=len(CONFIG.data.lr_variables),
            hr_shape=tuple(CONFIG.graph.hr_shape),
        ).to(DEVICE)
        try:
            skip.load_state_dict(ckpt["skip_block_state_dict"], strict=False)
            print(f"  [{name}] [+] skip_block ({skip.num_params()} params)")
        except Exception as e:
            print(f"  [{name}] [WARN] skip_block load failed: {e}")
            skip = None'''

if OLD_SKIP_LOAD in src_new:
    src_new = src_new.replace(OLD_SKIP_LOAD, NEW_SKIP_LOAD)
    print("[OK] skip_block load aussi protege")

nb["cells"][4]["source"] = [ln + "\n" for ln in src_new.split("\n")[:-1]] + [src_new.split("\n")[-1]]

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[OK] _safe_load helper integre dans build_stack")
print(f"     Verifie None + dict-like, strip prefixes, strict=False")
print(f"     {len(nb['cells'])} cellules")
