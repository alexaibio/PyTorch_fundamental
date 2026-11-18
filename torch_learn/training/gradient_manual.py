import torch
import torch.nn as nn
import torch.nn.functional as F


# A simple linear regression model: y = Wx + b
class MyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1) * 0.1)    # learnable (because Parameter)
        self.bias = nn.Parameter(torch.zeros(1))            # learnable

    def forward(self, x):
        return self.weight * x + self.bias


# Data: y = 3x + 2
x = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.])
y = 3 * x + 2

model = MyLinear()


############# TRAIN: both full-batch amd mini-batch are possible
lr = 0.01
batch_size = 4      # set batch_size=8 for full-batch training or batch_size=4 for mini-batch
num_epochs = 5

for epoch in range(num_epochs):
    # ---- Training ----
    model.train()       # set to a train mode - dropout etc is turned on

    # Shuffle data each epoch
    perm = torch.randperm(x.size(0))
    x_shuffled = x[perm]
    y_shuffled = y[perm]

    for i in range(0, x.size(0), batch_size):
        x_batch = x_shuffled[i : i + batch_size]
        y_batch = y_shuffled[i : i + batch_size]

        # Forward
        y_pred = model(x_batch)                 # because we implemented forward() method
        loss = F.mse_loss(y_pred, y_batch)

        # Backward: computes gradients of the loss with respect to all tensors that have requires_grad=True
        loss.backward()

        # Manual updates
        with torch.no_grad():       # I’m doing a manual update, don’t track gradients for this.
            model.weight -= lr * model.weight.grad
            model.bias -= lr * model.bias.grad

            # Zeroing gradients
            # By default, PyTorch accumulates gradients into .grad,
            # we can use it for gradient accumulation - when GPU memory is not enough
            # but here we anyway zero gradients at every mini-batch
            model.weight.grad.zero_()
            model.bias.grad.zero_()

        print(f"Epoch {epoch+1}: w={model.weight.item():.4f}, b={model.bias.item():.4f}, loss={loss.item():.4f}")

        # ---- Validation ----
        model.eval()
        # TBD



