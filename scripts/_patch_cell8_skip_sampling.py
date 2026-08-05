"""Fix Cell 8 skip-sampling bug — wrap recompute in if _need_sampling."""
import ast
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb_path = ROOT / "path_c_plus/scripts/st_cdgm_path_c_option_c_9node.ipynb"

raw = subprocess.check_output(
    ["git", "show", "79cbe11:path_c_plus/scripts/st_cdgm_path_c_option_c_9node.ipynb"],
    cwd=ROOT,
)
nb_good = json.loads(raw.decode("utf-8"))
src_good = "".join(nb_good["cells"][9]["source"])

start = src_good.index("    _eval_time = time.time() - _t0\n")
end = src_good.index("    # ------- 4. BS45 + BS44 : loop over 3 GCMs")
section = src_good[start:end]

# Body to run only when sampling needed: from torch.cat through BS43 skip-write
body_start = section.index("    _pred_mean = torch.cat(_all_means, dim=0).cpu()\n")
body = section[body_start:]

# Indent body by 4 spaces for nesting under else
indented_body = "".join(
    ("    " + line if line.strip() else line) for line in body.splitlines(keepends=True)
)

new_section = (
    "    _eval_time = time.time() - _t0\n\n"
    "    if not _need_sampling:\n"
    "        print(f\"    [BS30] metrics on disk -- skip recompute (torch.cat)\")\n"
    "        if _fv_path.exists():\n"
    "            try:\n"
    "                _fv_loaded = json.loads(_fv_path.read_text())\n"
    "                print(f\"    [BS30] cached RMSE={_fv_loaded.get('rmse')}, \"\n"
    "                      f\"Pearson={_fv_loaded.get('pearson_corr', {}).get('per_sample_avg')}\")\n"
    "            except Exception:\n"
    "                pass\n"
    "        print(f\"    [BS42] domain_metrics.json exists -- skip recompute\")\n"
    "        print(f\"    [BS43] eval_samples.npz exists -- skip recompute\")\n"
    "    else:\n"
    + indented_body
)

nb = json.loads(nb_path.read_text(encoding="utf-8"))
src = "".join(nb["cells"][9]["source"])

# Remove obsolete dummy-var block before sampling loop
obsolete = (
    "    if not _need_sampling:\n"
    "        # Set dummy variables so downstream blocks (which still run for BS44\n"
    "        # GCM eval) don't NameError. BS44 closures _build_inputs + _sample_once\n"
    "        # are defined above and remain valid.\n"
    "        _pred_mean = _pred_std = _targets = _pred_full = None\n"
    "        _mu_concat = None; _valid = None\n"
    "        _rmse = _mae = _spread = _corr_global = _corr_per_sample = float(\"nan\")\n"
    "        _f1 = {}; _rapsd_d = None; _intervention = []; _dag_avg = None\n"
    "        _mu_verdict = \"skipped_existing\"; _shortcut = {\"verdict\": \"skipped_existing\"}\n"
    "        _metrics_scope = \"skipped_existing\"; _eval_time = 0.0\n"
    "        _corr_per_sample_list = []\n\n"
)
src = src.replace(obsolete, "")

old_start = src.index("    _eval_time = time.time() - _t0\n")
old_end = src.index("    # ------- 4. BS45 + BS44 : loop over 3 GCMs")
src = src[:old_start] + new_section + src[old_end:]

ast.parse(src)
nb["cells"][9]["source"] = [src]
nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("OK — Cell 8 skip-sampling fix applied")
