import torch

x = torch.tensor([[1,2,3],[4,5,6]])
print(x.shape, '\n', x, end="\n\n")

# 扩展第 0 维
x_0 = x.unsqueeze(0)
print(x_0.shape, '\n',  x_0, end="\n\n")

# 扩展第 1 维
x_1 = x.unsqueeze(1)
print(x_1.shape, '\n',  x_1, end="\n\n")

# 扩展第 2维
x_2 = x.unsqueeze(2)
print(x_2.shape, '\n',  x_2, end="\n\n")

# squeeze 缩减tensor的大小为1的维度
x = torch.ones(1,1,3)
print(x.shape, '\n', x, end="\n\n")

y = x.squeeze(dim=0)
print(y.shape, '\n', y, end="\n\n")

z = x.squeeze()
print(z.shape, '\n', z, end="\n\n")
