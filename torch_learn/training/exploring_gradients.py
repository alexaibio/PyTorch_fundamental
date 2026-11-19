import torch
import torch.nn as nn


x = torch.randn(5, 3)                               # 5 vectors of dim 3
layer = nn.Linear(in_features=3, out_features=2)    # input vector [1 2 3], two neurons

# first forward pass
out = layer(x)
loss = (out ** 2).sum()     # emulation of Sum of Squared Errors. No target values - not real!

# compute gradients
loss.backward()                     # calculate gradients of loss for every parameter and save them to .grad
print('------ gradients after first backwards() --------')
print(layer.weight.grad)            # Inspect gradients
print(layer.weight.grad.norm())     # sum them up to one number to observer the changes
print(layer.bias.grad.norm())


# If not to zero gradients, they  accumulate by default
out = layer(x)              # one more forward pass
loss = (out ** 2).sum()
loss.backward()
print('---- Grads after second forward pass -----')
print(layer.weight.grad)
print(layer.weight.grad.norm())

# Zero gradients before the next backward pass
layer.zero_grad()

loss = (layer(x) ** 2).sum()
loss.backward()

print(layer.weight.grad.norm())  # Fresh gradients

