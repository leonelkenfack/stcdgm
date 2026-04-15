import torch

checkpoint_path = "models/st_cdgm_checkpoint.pth"
try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Check RCN cell state dict
    rcn_sd = checkpoint.get('rcn_cell_state_dict', {})
    dag_keys = [k for k in rcn_sd.keys() if 'A_dag' in k]
    
    if not dag_keys:
        print("A_dag not found in rcn_cell_state_dict. Available keys:", list(rcn_sd.keys()))
    else:
        for k in dag_keys:
            A = rcn_sd[k]
            print(f"Key: {k}")
            print(f"Shape: {A.shape}")
            print("Values:")
            print(A)
            
            # Check for sparsity (absolute value < 1e-4)
            zero_count = (torch.abs(A) < 1e-4).sum().item()
            total = A.numel()
            print(f"Sparsity (|x| < 1e-4): {zero_count}/{total} ({100.0 * zero_count / total:.2f}%)")
except Exception as e:
    print(f"Error: {e}")
