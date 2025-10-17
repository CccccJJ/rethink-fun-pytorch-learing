import torch
import time

print(f"cuda is available: {torch.cuda.is_available()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

size = 10000
A_cpu = torch.rand(size, size) # 默认在 CPU 上创建 tensor
B_cpu = torch.rand(size, size)

start_cpu = time.time()
C_cpu = torch.mm(A_cpu, B_cpu) # 矩阵乘法
end_cpu = time.time()
cpu_time = end_cpu - start_cpu

print(f"CPU time: {cpu_time:.6f} sec")

if torch.cuda.is_available():
    A_gpu = A_cpu.to(device)
    B_gpg = B_cpu.to(device)

    start_gpu = time.time()
    C_gpu = torch.mm(A_gpu, B_gpg)
    torch.cuda.synchronize()
    end_gpu=time.time()
    gpu_time = end_gpu - start_gpu

    print(f"GPU time: {gpu_time:.6f} sec")
else:
    print(f"GPU not available, skipping GPU test.")
