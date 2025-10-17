import torch

x = torch.randn(4,4)
print(x)

x = x.reshape(2,8)
print(x)

x = torch.tensor([[1,2,3],[4,5,6]])
x_reshape = x.reshape(3,2)
x_transpose = x.permute(1,0)
print(f"reshape: {x_reshape}")
print(f"permute: {x_transpose}")
