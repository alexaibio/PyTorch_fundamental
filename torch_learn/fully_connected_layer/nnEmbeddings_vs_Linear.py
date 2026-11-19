"""
Illustrate how Fully connected layer might be organized
- nn.Parameter
- nn.Embedding
- nn.Linear

out_features - number of neurons
in_features - each neuron has two inputs  x = [x_1, x_2]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearParameterLayer(nn.Module):
    def __init__(self, in_features=3, out_features=2):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6]
        ], dtype=torch.float32))

        self.bias = nn.Parameter(torch.tensor(
            [0.5, 0.6],
            dtype=torch.float32
        ))
        print(f"Linear Prama init weights: {self.weight}")

    def forward(self, x):
        return x @ self.weight + self.bias


class LinearLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_features=3, out_features=2, bias=False)  # weight initialized with Kaiming Uniform initialization
        print(f"Linear initial weights: {self.fc.weight}")

    def forward(self, x):
        return self.fc(x)   # automatically does x @ W^T


class LinearEmbeddingLayer(nn.Module):
    def __init__(self, num_embeddings=3, embedding_dim=2):
        super().__init__()
        # this is not weights! This is initial Embeddings!
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)    # weights are initialized from Gauss
        # bias is usually not used or initialized with zeros
        print(f"Embedding initial weights: {self.embedding.weight}")

    def forward(self, indices):
        return self.embedding(indices) + self.bias



torch.manual_seed(42)
############################### Diference between Linear and Embedding layers
linear_model = LinearParameterLayer()
embedding_model = LinearEmbeddingLayer()

# Make linear and Embedding models identical (copy weights).Dont use bias
embedding_model.embedding.weight.data = linear_model.weight.data.clone()


# Generate input data
x_dense = torch.tensor([
    [1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.],
    [10., 10., 10.]
])
out_linear = linear_model(x_dense)

# Linear layer gets input vector and output neuron values
print("\n\nLinear input:\n", x_dense)
print("Linear output:\n", out_linear)

x_indices = torch.tensor([0, 1])        # picks rows 0 and 1
out_embed  = embedding_model(x_indices)  # row lookup

# Embedding: get as an input embedding's indices and output embedding vector itself
print("\n\nEmbedding input (indices):\n", x_indices)
print("Embedding output:\n", out_embed)
print(" The End")
