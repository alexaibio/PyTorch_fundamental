import torch
import torch.nn as nn

x = torch.randn(32, 128)
layer = nn.Linear(128, 128)
out = layer(x)

loss = (out ** 2).sum()
loss.backward()

print(layer.weight.grad.norm())
