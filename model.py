import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from preprocessing import word_to_index, UNK

class CBOW(nn.Module):
  def __init__(self, vocab_size, num_classes, embedding_dim, hidden_dim):
    super(CBOW, self).__init__()

    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.linear1 = nn.Linear(embedding_dim, hidden_dim)
    self.activation_function1 = nn.ReLU()

    self.linear2 = nn.Linear(hidden_dim, num_classes)
    self.activation_function2 = nn.LogSoftmax()

  def forward(self, sentence):
    indices = torch.tensor(
      [word_to_index.get(word, UNK) for word in sentence],
      dtype=torch.long
    )
    embeds = sum(self.embeddings(indices)).view(1,-1)
    out = self.linear1(embeds)
    out = self.activation_function1(out)
    out = self.linear2(out)
    out = self.activation_function2(out)
    return out