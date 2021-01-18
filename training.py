import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from preprocessing import language_set, language_to_index, vocab, X_train, y_train, X_validation, y_validation, X_test, y_test
from model import CBOW

def train_epoch(epoch, model, sentences, languages, loss_function, optimizer):
  total_loss = 0
  progress_bar = tqdm(zip(sentences, languages), unit=" sentences", total=len(sentences), desc="Epoch {}".format(epoch))
  for sentence, language in progress_bar:
    optimizer.zero_grad()
    log_probs = model(sentence)
    loss = loss_function(log_probs, torch.tensor([language_to_index[language]], dtype=torch.long))
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    progress_bar.set_postfix(loss=total_loss)
  return total_loss

def evaluate(model, sentences, languages):
  n_correct = 0
  progress_bar = tqdm(zip(sentences, languages), unit=" sentences", total=len(sentences), desc="Evaluating")
  
  for sentence, language in progress_bar:
    log_probs = model(sentence)
    predicted = log_probs.argmax()
    if language_to_index[language] == predicted:
      n_correct += 1
    progress_bar.set_postfix(n_correct=n_correct)
  return n_correct / len(sentences)

def train(embedding_dimension, number_of_hidden_layers, hidden_layer_dimension, activation_function, number_of_training_epochs, loss_function_choice, optimizer_choice, learning_rate):
  train_losses = []
  train_accuracies = []
  validation_accuracies = []
  cbow = CBOW(
    vocab_size=len(vocab),
    num_classes=len(language_set),
    embedding_dim=embedding_dimension,
    hidden_dim=hidden_layer_dimension,
    number_of_hidden_layers=number_of_hidden_layers,
    activation_function=activation_function
  )

  for epoch in range(number_of_training_epochs):
    train_losses.append(
      train_epoch(
        epoch,
        cbow,
        X_train,
        y_train,
        loss_function=loss_function_choice(),
        optimizer=optimizer_choice(cbow.parameters(), lr=learning_rate)
      )
    )
    train_accuracies.append(evaluate(cbow, X_train, y_train))
    validation_accuracies.append(evaluate(cbow, X_validation, y_validation))

  print(f"Training accuracy: {evaluate(cbow, X_train, y_train)}")
  print(f"Validation accuracy: {evaluate(cbow, X_validation, y_validation)}")
  print(f"Test accuracy: {evaluate(cbow, X_test, y_test)}")

  return evaluate(cbow, X_validation, y_validation)