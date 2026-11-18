# Illustrate how Fully Connected Linear layer and its difference from Embedding layer

## Clarifies:
- nn.Parameter
- nn.Linear
- nn.Embedding


## nn.Parameter and nn.Linear
Firstly, nn.Parameter and nn.Linear approaches are equivalent. They have
- in_features - an input vector x = [x_1, x_2], it is feeded to each neuron
- out_features - number of neurons, each has one output

Inside, an input vector is multiplied to Weights matrix, producing an output vector. 
Weights might be learned via training process. 
The only difference is that
- with Parameter we have to do multiplication explicitelly in forward(): `return x @ self.weight + self.bias`
- with Linear it is enough: `return self.fc(x)`


## nn.Embedding
Then, nn.Embedding approach is different. 
It is not a linear prediction, but таблица обучаемых векторов.

Linear и Embedding решают две разные задачи, внутри обоих есть обучаемая матрица весов.
- Linear: посчитать выходы ленейного слоя через умножение входа на матрицу весов
- Embedding: какой обучаемый вектор соответствует объекту с определённым индексом?


For example
E = nn.Embedding(num_embeddings=3, embedding_dim=2)
E =
    0.1  0.5
    0.3  0.8
    0.7  0.2

Размерность (3, 2):
- num_embeddings=3 — количество векторов в таблице.
- embedding_dim=2 — размерность каждого вектора.

таким образом
    idx = torch.tensor([0, 2])

E[idx] просто достает два embeddings
    tensor([
        [0.1, 0.5],
        [0.7, 0.2]
    ])
здесь нет умножения входных значений на веса.

Но все же nn.Embedding содержит обучаемую матрицу весов, которая хранится в `embedding_model.embedding.weight.data` 
и которые тоже могут обучаться.

## Training Linear vs Embedding
In linear model, all weights participate in training, because every input feature is connected to every output neuron.

In Embedding model
- During the forward pass, PyTorch simply retrieves rows 0 and 2. There is no matrix multiplication.
- during backpropagation : 
  - Since only rows 0 and 2 were used in the forward pass, only these rows receive gradients from this operation.
  - The gradient for row 1 is zero for this batch because this embedding was not used.
  - The optimizer then updates the embedding vectors based on their gradients.
