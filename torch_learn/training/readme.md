# Manual and auto gradient update in PyTorch

We can update gradients in two ways in PyTorch
- manually - `model.weight -= lr * model.weight.grad`
- auto - `optimizer.step()`


## Autograd
Key points:
- Autograd is PyTorch’s automatic differentiation package.
- To compute all gradients we have to use backward(), for example loss.backward()
- then we can inspect gradients by looking at the `grad` attribute of a tensor.
- every time we use the gradients to update the parameters, we need to zero the gradients afterwards ( `zero_()` )

Exploring gradients - `exploring_gradients.py`

## Manual optimization

