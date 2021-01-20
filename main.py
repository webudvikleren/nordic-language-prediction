from training import train
import torch.nn as nn
import torch.optim as optim
import numpy as np
from hyperparameter_search import hyperparameter_search
import random
import matplotlib.pyplot as plt

'''
Introduction To Machine Learning: Hand-in 7

MIKKEL GODTFREDSEN (XRQ510)
SEBESTIAN ELIASSEN (FSP585)
MATHIAS GAMMELGAARD(PGW622)
JONATHAN CHRISTIANSEN (DVG554)
'''

#Task 2.1
print("Running Task 5.1...")
print("")
embedding_dim = random.randint(5, 20)
print(f"Embedding dim: {embedding_dim}")
print("")
train(**{
  "embedding_dimension": embedding_dim,
  "number_of_hidden_layers": 2,
  "hidden_layer_dimension": 2,
  "activation_function": nn.ReLU,
  "number_of_training_epochs": 10,
  "loss_function_choice": nn.NLLLoss,
  "optimizer_choice": optim.SGD,
  "learning_rate": 0.01
})

# Task 2.2
# Fidning the best params out of 15 combinations
print("Finding the best params out of 15 combinations...")
print("")
best_params = hyperparameter_search(15)[0][1]

# Finding the best accuracy of n = 1 to n=15 combinations in hyperparameter search
print("Finding the best accuracy of n = 1 to n=15 combinations in hyperparameter search")
print("")
best_models = []
for i in range(15):
  best_validation = hyperparameter_search(i+1)[0][1]
  best_models.append(best_validation)
plt.title("Validation accuracy as a function of combinations tested")
plt.xticks(np.arange(1, 16))
plt.ylabel('best validation accuracy')
plt.xlabel('number of hyperparameter combinations')
plt.plot(np.arange(1, 16), best_models)
plt.show()

#Task 2.3
# Finding training- and validation accuracies pr. epoch for the model with best params..
print("Finding training- and validation accuracies pr. epoch for the model with best params...")
print("")
train_accuracy, validation_accuracy, test_accuracy, train_accuracies, validation_accuracies = train(**best_params)

train_loss = 1 - np.array(train_accuracies)
X = range(1, 11)
plt.plot(X, train_accuracies, label='train acc')
plt.plot(X, validation_accuracies, label='val acc')
plt.plot(X, train_loss, label='train loss')
plt.title('Training acc, validation acc and training loss')
plt.xlabel('Number of epochs')
plt.ylabel('')
plt.legend()
plt.show()

#Task 2.4
best_params["use_rnn"] = True
train(**best_params)

#Task 2.5
best_params["use_LSTM"] = True
train(**best_params)

best_params["use_LSTM"] = False
best_params["use_GRU"] = True

train(**best_params)