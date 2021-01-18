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
#   "learning_rate": 0.001
# })

#Task 5.2
hyperparameter_search(10)
