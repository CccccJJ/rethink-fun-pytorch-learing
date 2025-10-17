import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputs = torch.tensor([[2, 1000], [3, 2000], [2, 500], [1, 800], [4, 3000]], dtype=torch.float, device=device)
labels = torch.tensor([[19], [31], [14], [15], [43]], dtype=torch.float, device=device)

# 归一化
# inputs = inputs / torch.tensor([4, 3000], device=device)

# 标准化
mean = inputs.mean(dim=0)
std = inputs.std(dim=0)
inputs = (inputs - mean) / std

w = torch.ones((2, 1), requires_grad=True, device=device)
b = torch.ones((1, ), requires_grad=True, device=device)

epoch = 1000
lr = 0.5

for i in range(epoch):
    outputs = inputs @ w + b
    loss = torch.mean(torch.square(outputs - labels))
    
    loss.backward()
    
    if i % 10 == 0:
        print(f"loss: {loss.item()}")
        print(f"w.grad: {w.grad.tolist()}")
    
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        
    w.grad.zero_()
    b.grad.zero_()
    
print(f"Final w: {w}, b: {b}", end="\n\n")

# 预测
test_x = torch.tensor([[2, 1000]], dtype=torch.float, device=device)
test_label = torch.tensor(19, dtype=torch.float, device=device)

# 归一化处理
# test_x = test_x / torch.tensor([4, 3000], dtype=torch.float, device=device)

# 标准化
test_x = (test_x - mean) / std

test_y = test_x @ w  + b
print(f"test_y: {test_y.item()}, test_label: {test_label}")
