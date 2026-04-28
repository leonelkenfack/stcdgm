import json

file_path = 'st_cdgm_training_evaluation.ipynb'
cell_index = 52

with open(file_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

source = notebook['cells'][cell_index]['source']

# Define the new lines for the cell
new_source = []
skip = False
for line in source:
    if 'encoder_target =' in line and not skip:
        new_source.append("def get_core_model(m):\n")
        new_source.append("    m = m.module if hasattr(m, \"module\") else m\n")
        new_source.append("    return getattr(m, \"_orig_mod\", m)\n")
        new_source.append("\n")
        new_source.append("encoder_target = get_core_model(encoder)\n")
        new_source.append("rcn_target = get_core_model(rcn_cell)\n")
        new_source.append("diff_target = get_core_model(diffusion)\n")
        skip = True
    elif 'rcn_target =' in line or 'diff_target =' in line:
        continue
    else:
        new_source.append(line)

notebook['cells'][cell_index]['source'] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print(f"Cell {cell_index} updated successfully.")
