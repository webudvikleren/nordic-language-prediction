import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from preprocessing import word_to_index, UNK
import collections

class CBOW(nn.Module):
  def __init__(self, vocab_size, num_classes, embedding_dim, hidden_dim, number_of_hidden_layers, activation_function):
    super(CBOW, self).__init__()

    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.number_of_hidden_layers = number_of_hidden_layers

    #Dynamically set hidden layers, with given activation function
    for i in range(1,number_of_hidden_layers+1):
      if i == 1:
        setattr(self, f"linear{i}", nn.Linear(embedding_dim, hidden_dim) )
        setattr(self, f"activation_function{i}", activation_function())
      elif i == number_of_hidden_layers:
        setattr(self, f"linear{i}", nn.Linear(hidden_dim, num_classes) )
        setattr(self, f"activation_function{i}", nn.LogSoftmax())
      else:
        setattr( self, f"linear{i}", nn.Linear(hidden_dim, hidden_dim) )
        setattr(self, f"activation_function{i}", activation_function())

  def forward(self, sentence):
    indices = torch.tensor(
      [word_to_index.get(word, UNK) for word in sentence],
      dtype=torch.long
    )
    embeds = sum(self.embeddings(indices)).view(1,-1)

    #Dynamically calculating predictions
    out = None
    for i in range(1, self.number_of_hidden_layers + 1):
      if i == 1:
        out = getattr(self, f"linear{i}")(embeds)
      else:
        out = getattr(self, f"linear{i}")(out)
      out = getattr(self, f"activation_function{i}")(out)
    return out