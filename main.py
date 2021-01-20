from training import train
import torch.nn as nn
import torch.optim as optim
import numpy as np
from hyperparameter_search import hyperparameter_search
import random

#Task 5.1
# embedding_dim = random.randint(5, 20)
# print(f"Embedding dim: {embedding_dim}")
# print("")
# train(**{
#   "embedding_dimension": embedding_dim,
#   "number_of_hidden_layers": 2,
#   "hidden_layer_dimension": 2,
#   "activation_function": nn.ReLU,
#   "number_of_training_epochs": 10,
#   "loss_function_choice": nn.NLLLoss,
#   "optimizer_choice": optim.SGD,
#   "learning_rate": 0.01
# })

#Task 5.2
# hyperparameter_search(10)

#Task 5.3
best_params = {
  'embedding_dimension': 32,
  'number_of_hidden_layers': 5,
  'hidden_layer_dimension': 32,
  'activation_function': nn.Sigmoid,
  'number_of_training_epochs': 10,
  'loss_function_choice': nn.NLLLoss,
  'optimizer_choice': optim.Adam,
  'learning_rate': 0.01
}

train(**best_params)
