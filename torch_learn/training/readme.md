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
We can organize the learning loop completely manually, i.e update parameters directly without optimizer.

Look at `gradient_manual.py`


## Use gradient optimizer
Alternatively we can use Optimizer

Look at `gradient_optimizer.py`

The difference lies in this block. Instead of updating parameters manually

```python
with torch.no_grad():
    model.weight -= lr * model.weight.grad
    model.bias -= lr * model.bias.grad

```
 we do it with just one line
`optimizer.step()`

There are many different Optimizers and they can use different schemas of updating parameters, not only 
a simple one like Gradient Descent.
Among them
- SGD с Momentum
- Adam
- AdamW

