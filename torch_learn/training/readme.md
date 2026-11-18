# Manual and auto gradient update in PyTorch

We can update gradients in two ways in PyTorch
- manually - `model.weight -= lr * model.weight.grad`
- auto - `optimizer.step()`