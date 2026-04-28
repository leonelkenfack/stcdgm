import json
import sys

file_path = 'st_cdgm_training_evaluation.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Old code lines as they appear in the JSON
old_lines = [
    "    \"encoder_target = encoder.module if hasattr(encoder, \\\"module\\\") else encoder\\n\",\n",
    "    \"rcn_target = rcn_cell.module if hasattr(rcn_cell, \\\"module\\\") else rcn_cell\\n\",\n",
    "    \"diff_target = diffusion.module if hasattr(diffusion, \\\"module\\\") else diffusion\\n\",\n",
    "    \"\\n\",\n",
    "    \"try:\\n\",\n",
    "    \"    from st_cdgm.utils.checkpoint import strip_torch_compile_prefix\\n\",\n",
    "    \"except ModuleNotFoundError:\\n\",\n",
    "    \"    def strip_torch_compile_prefix(sd):\\n\",\n",
    "    \"        if not sd:\\n\",\n",
    "    \"            return sd\\n\",\n",
    "    \"        if not any(str(k).startswith(\\\"_orig_mod.\\\") for k in sd.keys()):\\n\",\n",
    "    \"            return sd\\n\",\n",
    "    \"        return {\\n\",\n",
    "    \"            (k[len(\\\"_orig_mod.\\\") :] if str(k).startswith(\\\"_orig_mod.\\\") else k): v\\n\",\n",
    "    \"            for k, v in sd.items()\\n\",\n",
    "    \"        }\\n\",\n",
    "    \"\\n\",\n",
    "    \"encoder_target.load_state_dict(strip_torch_compile_prefix(loaded_ckpt[\\\"encoder_state_dict\\\"]))\\n\",\n",
    "    \"rcn_target.load_state_dict(strip_torch_compile_prefix(loaded_ckpt[\\\"rcn_cell_state_dict\\\"]))\\n\",\n",
    "    \"diff_target.load_state_dict(strip_torch_compile_prefix(loaded_ckpt[\\\"diffusion_state_dict\\\"]))\\n\""
]

# New code lines
new_lines = [
    "    \"def get_core_model(m):\\n\",\n",
    "    \"    m = m.module if hasattr(m, \\\"module\\\") else m\\n\",\n",
    "    \"    return getattr(m, \\\"_orig_mod\\\", m)\\n\",\n",
    "    \"\\n\",\n",
    "    \"encoder_target = get_core_model(encoder)\\n\",\n",
    "    \"rcn_target = get_core_model(rcn_cell)\\n\",\n",
    "    \"diff_target = get_core_model(diffusion)\\n\",\n",
    "    \"\\n\",\n",
    "    \"try:\\n\",\n",
    "    \"    from st_cdgm.utils.checkpoint import strip_torch_compile_prefix\\n\",\n",
    "    \"except ModuleNotFoundError:\\n\",\n",
    "    \"    def strip_torch_compile_prefix(sd):\\n\",\n",
    "    \"        if not sd:\\n\",\n",
    "    \"            return sd\\n\",\n",
    "    \"        if not any(str(k).startswith(\\\"_orig_mod.\\\") for k in sd.keys()):\\n\",\n",
    "    \"            return sd\\n\",\n",
    "    \"        return {\\n\",\n",
    "    \"            (k[len(\\\"_orig_mod.\\\") :] if str(k).startswith(\\\"_orig_mod.\\\") else k): v\\n\",\n",
    "    \"            for k, v in sd.items()\\n\",\n",
    "    \"        }\\n\",\n",
    "    \"\\n\",\n",
    "    \"encoder_target.load_state_dict(strip_torch_compile_prefix(loaded_ckpt[\\\"encoder_state_dict\\\"]))\\n\",\n",
    "    \"rcn_target.load_state_dict(strip_torch_compile_prefix(loaded_ckpt[\\\"rcn_cell_state_dict\\\"]))\\n\",\n",
    "    \"diff_target.load_state_dict(strip_torch_compile_prefix(loaded_ckpt[\\\"diffusion_state_dict\\\"]))\\n\""
]

modified = False
for cell in notebook['cells']:
    if 'source' in cell:
        source = cell['source']
        # Check if the sequence of lines exists in this cell
        for i in range(len(source) - len(old_lines) + 1):
            if source[i:i+len(old_lines)] == old_lines:
                source[i:i+len(old_lines)] = new_lines
                modified = True
                break
        if modified:
            break

if modified:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    print("Notebook updated successfully.")
else:
    print("Could not find the target code block in the notebook.")
