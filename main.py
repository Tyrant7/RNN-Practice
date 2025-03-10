import torch

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
torch.set_default_device(device)

print(f"Using device = {torch.get_default_Device()}")