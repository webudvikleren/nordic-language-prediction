import torch.nn as nn
import torch.optim as optim
from numpy.random import choice
from training import train

def hyperparameter_search( number_of_combinations ):
  embedding_dimension = [ 2, 16, 64, 128 ]
  number_of_hidden_layers = [ 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 ]
  hidden_layer_dimension = [ 2, 16, 64, 128 ]
  activation_function = [ nn.ReLU, nn.Sigmoid, nn.Tanh ]
  number_of_training_epochs = [ 5, 10, 20 ]
  loss_function_choice = [ nn.NLLLoss, nn.L1Loss, nn.CrossEntropyLoss, nn.MSELoss ]
  optimizer_choice = [ optim.SGD, optim.Adam ]
  learning_rate = [ 0.001, 0.01, 0.1, 1 ]

  params_results = []

  #TODO - Memorize each choice of params, so no combination is selected more than once

  for i in range(number_of_combinations):
    params = {
      "embedding_dimension": choice(embedding_dimension),
      "number_of_hidden_layers": choice(number_of_hidden_layers),
      "hidden_layer_dimension": choice(hidden_layer_dimension),
      "activation_function": choice(activation_function),
      "number_of_training_epochs": choice(number_of_training_epochs),
      "loss_function_choice": choice(loss_function_choice),
      "optimizer_choice": choice(optimizer_choice),
      "learning_rate": choice(learning_rate)
    }
    print("Searching with following params:s")
    print(params)
    validation_accuracy = train(**params)
    print("")
    params_results.append(( params, validation_accuracy ))

  params_results.sort(key=lambda x: x[1])
  print("Best found params:")
  print(params_results[0][0])
  print("")
  print("Accuracy on best found params")
  print(params_results[0][1])
  print("")
    
  return params_results