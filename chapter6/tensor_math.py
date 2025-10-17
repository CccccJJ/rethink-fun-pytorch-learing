import torch

a = torch.ones(2,3)
b = torch.ones(2,3)

print(a + b,end="\n\n")
print(a - b,end="\n\n")
print(a * b,end="\n\n")
print(a / b,end="\n\n")
print(a @ b.t(),end="\n\n") # 矩阵乘法

