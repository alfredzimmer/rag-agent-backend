import torch

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Supported Archs: {torch.cuda.get_arch_list()}")

# if supported archs include sm_120, then CUDA is available for 5090