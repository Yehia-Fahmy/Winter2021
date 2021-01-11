import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np
import time as t


def load_data():        # brings in the data that we saved as a pickle file
    print("Loading data...")
    X = pickle.load(open("X.pickle", "rb"))
    y = pickle.load(open("y.pickle", "rb"))
    return X, y


def reshape_data(X, y):
    print("Reshaping data...")
    X = np.array(X)     # ensuring that lists are instead arrays
    X = X / 255         # normalizing the data
    y = np.array(y)
    return X, y


def build_network(X, y):
    print("Building network...")

    model = Sequential()
    for i in range(numlayers):      # adds a layer
        model.add(Conv2D(numnodes, (3, 3), input_shape=X.shape[1:]))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation("sigmoid"))        # the final layer is responsible for the prediction

    print("Compiling model...")
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])     # compiles the model

    return model


def train_model(model, X, y):   # this function is responsible for training the model
    print("Training model...")
    numepochs = 3               # number of epochs we want to train for
    batchsize = 32              # higher batch size will train faster
    trained_model = model.fit(X, y, epochs=numepochs, validation_split=0.1, batch_size= batchsize)
    return trained_model


numlayers = 2       # number of layers in the network
numnodes = 64       # number of nodes in each layer

NAME = f"insulator_classifier_{numnodes}x{numlayers}_{int(t.time())}"
print(NAME)

start_time = t.time()
print("Starting...")

X, y = load_data()
X, y = reshape_data(X, y)

our_model = build_network(X, y)
new_model = train_model(our_model, X, y)

total_time = t.time() - start_time
total_time = round(total_time, 2)
print("Finished in: ", total_time, "s!")
