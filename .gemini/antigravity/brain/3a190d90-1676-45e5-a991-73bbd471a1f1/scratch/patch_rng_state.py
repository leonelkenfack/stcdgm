import json

file_path = 'st_cdgm_training_evaluation.ipynb'
cell_index = 44

with open(file_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

source = notebook['cells'][cell_index]['source']

new_source = []
for line in source:
    if 'torch.set_rng_state(ckpt["rng_torch_cpu"])' in line:
        # Get the indentation
        indent = line[:line.find('torch.set_rng_state')]
        new_source.append(f"{indent}try:\n")
        new_source.append(f"{indent}    torch.set_rng_state(ckpt['rng_torch_cpu'])\n")
        new_source.append(f"{indent}except TypeError:\n")
        new_source.append(f"{indent}    # Fallback: cast to ByteTensor (uint8) if needed\n")
        new_source.append(f"{indent}    torch.set_rng_state(torch.as_tensor(ckpt['rng_torch_cpu'], dtype=torch.uint8).cpu())\n")
    elif 'torch.cuda.set_rng_state_all(ckpt["rng_torch_cuda"])' in line:
        indent = line[:line.find('torch.cuda.set_rng_state_all')]
        new_source.append(f"{indent}try:\n")
        new_source.append(f"{indent}    torch.cuda.set_rng_state_all(ckpt['rng_torch_cuda'])\n")
        new_source.append(f"{indent}except (TypeError, RuntimeError):\n")
        new_source.append(f"{indent}    # Fallback pour CUDA\n")
        new_source.append(f"{indent}    pass\n")
    else:
        new_source.append(line)

notebook['cells'][cell_index]['source'] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print(f"Cell {cell_index} patched successfully.")
