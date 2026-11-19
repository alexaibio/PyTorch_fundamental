import torch
import torch.nn as nn
import torch.nn.functional as F


# Linear regression model, one dimensional input : y = Wx + b
class MyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1) * 0.1)    # x is one dimensional
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.weight * x + self.bias


model = MyLinear()

############# Training Data: y = 3x + 2
x = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.])
y = 3 * x + 2

############# TRAINING with batches
lr = 0.01
batch_size = 4      # set batch_size=8 for full-batch training or batch_size=4 for mini-batch
num_epochs = 5

# Epoch loop, go through the same data several times
for epoch in range(num_epochs):
    model.train()       # set to a train mode - dropout etc are taken into account

    # Shuffle data each epoch
    perm = torch.randperm(x.size(0))
    x_shuffled = x[perm]
    y_shuffled = y[perm]

    epoch_loss = 0.0

    # Training loop (by batches)
    for i_batch_start in range(0, x.size(0), batch_size):
        x_batch = x_shuffled[i_batch_start : i_batch_start + batch_size]
        y_batch = y_shuffled[i_batch_start : i_batch_start + batch_size]

        # Forward
        y_pred = model(x_batch)
        loss = F.mse_loss(y_pred, y_batch)      #Mean Squared Error (MSE)

        model.zero_grad()   # zeroing grads
        loss.backward()     # Backward: computes gradients of the loss with respect to all tensors that have requires_grad=True

        # Manual parameter update
        with torch.no_grad():           # Don't track operations inside this block for automatic differentiation.
            model.weight -= lr * model.weight.grad
            model.bias -= lr * model.bias.grad

    epoch_loss += loss.item() * x_batch.size(0)

    print(
        f"Epoch {epoch+1}: "
        f"w={model.weight.item():.4f}, "
        f"b={model.bias.item():.4f}, "
        f"loss={epoch_loss:.4f}"
    )


# ---- Validation ----
model.eval()
with torch.no_grad():
    predictions = model(x)
    final_loss = F.mse_loss(predictions, y)

print("\nFinal results:")
print("Weight:", model.weight.item())
print("Bias:", model.bias.item())
print("Final loss:", final_loss.item())



