from tensorflow import keras
import numpy as np
import pickle
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix


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
    # y_test = to_categorical(y_test)
    X_test = X_test / 255  # normalizing the data
    return X_test, y_test


NAME = "mark 4.40 attempt 3 (3 epochs)"
our_model = load_model(NAME)
X_test, y_test = load_data()

Y_pred = our_model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
