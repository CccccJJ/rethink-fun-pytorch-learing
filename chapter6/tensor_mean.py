import torch

t = torch.tensor([[1.0,  3.0], [1.0, 3.0],[1.0, 3.0]])

mean = t.mean()
print(f"mean: {mean}")

mean = t.mean(dim=0)
print(f"mean on dim 0: {mean}")

mean = t.mean(dim=1)
print(f"mean on dim 1: {mean}")
