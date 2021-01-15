import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("NordicDSL/data/wikipedia/train.csv")

X = df["sentence"].values
y = df["language"].values

X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.01, test_size=0.001)
df = pd.read_csv("NordicDSL/data/wikipedia/test.csv")
X_test = df["sentence"].values
y_test = df["language"].values

def tokenize(sentences):
  return [sentence.split() for sentence in sentences]

X_train = tokenize(X_train)
X_validation = tokenize(X_validation)
X_test = tokenize(X_test)

vocab = set().union(*X_train) | {"UNK"}
word_to_index = {word: i for i, word in enumerate(vocab)}
UNK = word_to_index["UNK"]

language_set = set(y_train)
language_to_index = {language: i for i, language in enumerate(language_set)}