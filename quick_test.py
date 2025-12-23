import torch

checkpoint = torch.load("checkpoints/convnext_base_best.pth", weights_only=False)
print("SUCCESS! Checkpoint loaded.")
print("Keys:", list(checkpoint.keys()))