import torch
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inputs = torch.rand(100, 3)
weights = torch.tensor([[1.1], [2.2], [3.3]])
bias = torch.tensor(4.4)
targets = inputs @ weights + bias + 0.1 * torch.randn(100, 1)

# tensorboard --logdir=./lr/runs/
writer = SummaryWriter(log_dir="./lr/runs/")

w = torch.rand((3, 1), requires_grad=True, device=device)
b = torch.rand((1,), requires_grad=True, device=device)

inputs = inputs.to(device)
targets = targets.to(device)

epoch = 10000
lr = 0.003

for i in range(epoch):
    outputs = inputs @ w + b
    loss = torch.mean(torch.square(outputs - targets))
    
    writer.add_scalar("loss/train", loss.item(), i)

    if i % 1000 == 0:
        print(f"loss: {loss.item()}")

    loss.backward()
    
    with torch.no_grad(): # 接下来的计算不需要跟踪梯度
        w -= lr * w.grad
        b -= lr * b.grad
    
    # 清零梯度
    w.grad.zero_()
    b.grad.zero_()

print(f"训练后的权重 w: {w}")
print(f"训练后的偏执 b: {b}")
