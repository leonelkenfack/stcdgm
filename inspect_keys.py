import torch

checkpoint_path = "models/st_cdgm_checkpoint.pth"
try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    rcn_sd = checkpoint.get('rcn_cell_state_dict', {})
    print("Keys in rcn_cell_state_dict:")
    for k in sorted(rcn_sd.keys()):
        print(f"  {k}: {rcn_sd[k].shape}")
        
    encoder_sd = checkpoint.get('encoder_state_dict', {})
    print("\nKeys in encoder_state_dict:")
    for k in sorted(encoder_sd.keys()):
        print(f"  {k}: {encoder_sd[k].shape if hasattr(encoder_sd[k], 'shape') else 'no shape'}")
except Exception as e:
    print(f"Error: {e}")
