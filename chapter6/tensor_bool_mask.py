import torch

x = torch.tensor([1,2,3,4,5])
mask = x > 2
print(mask)

filtered_x = x[mask]
print(filtered_x)

x[mask] = 0
print(x)
