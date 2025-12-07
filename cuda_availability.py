import torch
print("CUDA available:", torch.cuda.is_available())
print("Torch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
