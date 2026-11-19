import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Linear regression model, one dimensional input : y = Wx + b
class MyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.weight * x + self.bias


# Model + optimizer
lr = 0.01
model = MyLinear()
optimizer = optim.SGD(model.parameters(), lr=lr)


# Data
x = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.])
y = 3 * x + 2


# TRAINING
batch_size = 4
num_epochs = 10

for epoch in range(num_epochs):
    model.train()

    # Shuffle data each epoch
    perm = torch.randperm(x.size(0))
    x_shuffled = x[perm]
    y_shuffled = y[perm]

    # Mini-batch loop
    for i_batch_start in range(0, x.size(0), batch_size):
        x_batch = x_shuffled[i_batch_start : i_batch_start + batch_size]
        y_batch = y_shuffled[i_batch_start : i_batch_start + batch_size]

        # Forward
        y_pred = model(x_batch)
        loss = F.mse_loss(y_pred, y_batch)

        # Backward + optimizer step
        optimizer.zero_grad()   # zero grads for this batch
        loss.backward()
        optimizer.step()        # вместо целого блока update parameters with torch.no_grad()

    # Print progress at end of epoch
    with torch.no_grad():
        y_pred_full = model(x)
        loss_full = F.mse_loss(y_pred_full, y)
        print(f"Epoch {epoch+1}: w={model.weight.item():.4f}, "
              f"b={model.bias.item():.4f}, loss={loss_full.item():.4f}")
