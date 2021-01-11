from tensorflow import keras
import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical


def load_model(name):
    print(f"Loading {name}...")
    model = keras.models.load_model(name)
    model.summary()
    return model


def load_data():
    print("loading data...")
    X_test = pickle.load(open("X_test.pickle", "rb"))
    y_test = pickle.load(open("y_test.pickle", "rb"))
    return X_test, y_test


def reshape_data(X_test, y_test):
    print("Reshaping data...")
    X_test = np.array(X_test)
    y_test = np.array(y_test)           # ensuring the lists are numpy arrays
    y_test = to_categorical(y_test)
    X_test = X_test / 255               # normalizing the data
    return X_test, y_test


def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=16, use_multiprocessing='True')
    test_acc = round(test_acc * 100, 2)
    return test_acc


X_test, y_test = load_data()
X_test, y_test = reshape_data(X_test, y_test)
x = 5
accuracies = np.zeros(x)

for i in range(x):
    name = f'Rectangle attempt {i} (3 epochs)'
    our_model = load_model(name)
    accuracies[i] = evaluate_model(our_model, X_test, y_test)

acc = np.sum(accuracies) / x

avg = round(acc, 2)
max = np.max(accuracies)
min = np.min(accuracies)
dev = np.std(accuracies)
print(f"Rectangle avg: {avg}%, max: {max}, dev: {dev}")