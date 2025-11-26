import os
import torch

print("|" + "-"*100 + "|")
print(f"| CUDA Available: {torch.cuda.is_available()}")
print(f"| PyTorch Version: {torch.__version__}")
print(f"| CUDA Version: {torch.version.cuda}")
print(f"| Supported Archs: {torch.cuda.get_arch_list()}")

assert torch.cuda.is_available(), "CUDA is not available"
assert "sm_120" in torch.cuda.get_arch_list(), "CUDA is not available for RTX 5090"

torch.cuda.set_per_process_memory_fraction(float(os.getenv("CUDA_MEMORY_FRACTION") or "0.5"), device=0)
print(f"| Using CUDA Memory Fraction: {torch.cuda.get_per_process_memory_fraction(device=0)}")
print("|" + "-"*100 + "|")

