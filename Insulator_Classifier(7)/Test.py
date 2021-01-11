import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from tensorflow import keras
from tensorflow.keras.utils import to_categorical


def load_model(name):
    print(f"Loading {name}...")
    model = keras.models.load_model(name)
    return model


def load_data():
    print("loading data...")
    X_test = pickle.load(open("X_test.pickle", "rb"))
    y_test = pickle.load(open("y_test.pickle", "rb"))
    print("Reshaping data...")
    X_test = np.array(X_test)
    y_test = np.array(y_test)  # ensuring the lists are numpy arrays
    y_test = to_categorical(y_test)
    X_test = X_test / 255  # normalizing the data
    return X_test, y_test


def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=16, use_multiprocessing='True')
    test_acc = round(test_acc * 100, 2)
    return test_acc


X_test, y_test = load_data()
max_acc = 0
max_model = ''

name = f"mark 4.40 attempt 1 (3 epochs)"
our_model = load_model(name)
our_model.summary()


'''for i in range(5):
    name = f"mark 4.40 attempt {i} (3 epochs)"
    our_model = load_model(name)
    temp = evaluate_model(our_model, X_test, y_test)
    if temp > max_acc:
        max_acc = temp
        max_model = name'''

print(f"{max_model} has acc of: {max_acc}")