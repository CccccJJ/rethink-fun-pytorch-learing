import torch

shape = (2,3)

rand_tensor = torch.rand(shape) # [0,1] 均匀抽样
randn_tensor = torch.randn(shape) # 标准正态分布
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
twos_tensor = torch.full(shape,2)

print(rand_tensor)
print(randn_tensor)
print(ones_tensor)
print(zeros_tensor)
print(twos_tensor)
