import torch

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()

loss = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

for epoch in range(1000):
    y_pred = model.forward(x_data)
    loss_data = loss(y_pred, y_data)
    print(epoch, loss_data)

    optimizer.zero_grad()
    loss_data.backward()
    optimizer.step()

# 预测
y = model.forward(torch.Tensor([[4.0]]))
print(y)