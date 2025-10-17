import torch

t1 = torch.randn((3,2))
print(t1)

t2 = t1 + 1
print(t2)

t1 = torch.ones((3,2))
print(t1)
t2 = torch.ones(2)
print(t2)

t3 = t1 + t2
print(t3)
