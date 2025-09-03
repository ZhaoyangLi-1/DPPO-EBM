"""Quick script to check EBM checkpoint structure."""
import torch

# Check checkpoint structure
ckpt_path = "/linting-slow-vol/EBM-Guidance/outputs/gym/ebm_best_val_hopper-v2-medium.pt"
checkpoint = torch.load(ckpt_path, map_location='cpu')

print("Checkpoint keys:", list(checkpoint.keys()))
if isinstance(checkpoint, dict):
    for key, value in checkpoint.items():
        if hasattr(value, 'keys'):
            print(f"  {key}: {list(value.keys())}")
        else:
            print(f"  {key}: {type(value)}")