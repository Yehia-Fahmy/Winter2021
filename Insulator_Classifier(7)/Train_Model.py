from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np
import time as t


def load_data():                 # brings in the data that we saved as a pickle file
    print("Loading data...")
    y_train = pickle.load(open("y_train.pickle", "rb"))
    X_test = pickle.load(open("X_test.pickle", "rb"))
    y_test = pickle.load(open("y_test.pickle", "rb"))
    X_train = pickle.load(open("X_train.pickle", "rb"))
    return X_train, y_train, X_test, y_test


def reshape_data(X_train, y_train, X_test, y_test):
    print("Reshaping data...")
    X_train = np.array(X_train)
    y_train = np.array(y_train)         # ensuring the lists are numpy arrays
    y_train = to_categorical(y_train)
    X_train = X_train / 255             # normalizing the data

    X_test = np.array(X_test)
    y_test = np.array(y_test)           # ensuring the lists are numpy arrays
    y_test = to_categorical(y_test)
    X_test = X_test / 255               # normalizing the data
    return X_train, y_train, X_test, y_test


def build_network(X):
    print("Building network...")
    model = Sequential()

    for i in range(LAYERS):      # adds a layer
        model.add(Conv2D(NODES, kernel_size=(4, 4), activation='relu', input_shape=(X.shape[1:])))
        # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(7, activation='softmax'))

    print("Compiling network...")
    model.compile(loss='categorical_crossentropy',
                  optimizer='adamax',
                  metrics=['accuracy'])
    model.summary()
    print(f"Optimizer Used: adamax")
    return model


def build_YehiaNet(X):
    print("Building YehiaNet...")
    model = Sequential()
    # First Layer
    model.add(Conv2D(512, kernel_size=(4, 4), activation='relu', input_shape=(X.shape[1:]), padding='same', strides=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Second Layer
    model.add(Conv2D(512, kernel_size=(4, 4), activation='relu', padding='same', strides=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Fourth Layer
    model.add(Conv2D(512, kernel_size=(4, 4), activation='relu', padding='same', strides=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Fifth Layer
    model.add(Conv2D(512, kernel_size=(4, 4), activation='relu', padding='same', strides=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Sixth Layer
    model.add(Flatten())
    model.add(Dense(7, activation='softmax'))
    print("Compiling network...")
    model.compile(loss='categorical_crossentropy',
                  optimizer='adamax',
                  metrics=['accuracy'])
    model.summary()
    return model


def build_rectangle_dense(X, num_layers, num_nodes):
    print("Building Rectangle dense")
    model = Sequential()
    for i in range(num_layers):
        model.add(Dense(activation='relu', input_shape=(X.shape[1:]), units=num_nodes))

    model.add(Flatten())
    model.add(Dense(7, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adamax',
                  metrics=['accuracy'])
    model.summary()
    return model


def build_Mark_4_40(X):
    print("Building Mark 4.40...")
    model = Sequential()
    # First Layer
    model.add(
        Conv2D(512, kernel_size=(4, 4), activation='relu', input_shape=(X.shape[1:]), padding='same', strides=(2, 2)))
    # Second Layer
    model.add(Conv2D(256, kernel_size=(4, 4), activation='relu', padding='same', strides=(2, 2)))
    # Third Layer
    model.add(Conv2D(128, kernel_size=(4, 4), activation='relu', padding='same', strides=(2, 2)))
    # Fourth Layer
    model.add(Conv2D(128, kernel_size=(4, 4), activation='relu', padding='same', strides=(2, 2)))
    # Fifth Layer
    model.add(Conv2D(256, kernel_size=(4, 4), activation='relu', padding='same', strides=(2, 2)))
    # Sixth Layer
    model.add(Conv2D(512, kernel_size=(4, 4), activation='relu', padding='same', strides=(2, 2)))
    # Final Layer
    model.add(Flatten())
    model.add(Dense(7, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adamax',
                  metrics=['accuracy'])
    # All Done
    model.summary()
    return model


def train_network(model, X, y):
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, use_multiprocessing='True')


def evaluate_network(model, X, y):
    test_loss, test_acc = model.evaluate(X, y, batch_size=BATCH_SIZE, use_multiprocessing='True')
    test_acc = round(test_acc*100, 2)
    return test_acc


def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)


start_time = t.time()
# global variables
PATH = "/Users/Yehia/OneDrive - University of Waterloo/Summer 2020 Co-op/Saved_Models/Yehia_Net"
EPOCHS = 3
BATCH_SIZE = 80
NODES = 512
LAYERS = 4

print("Starting...")

# loading and reshaping the data
X_train, y_train, X_test, y_test = load_data()
X_train, y_train, X_test, y_test = reshape_data(X_train, y_train, X_test, y_test)


name = f"Rectangle dense attempt 0 ({EPOCHS} epochs)"
our_model = build_rectangle_dense(X_train, LAYERS, NODES)
train_network(our_model, X_train, y_train)
our_model.save(name)


total_time = t.time() - start_time
total_time = round(total_time, 2)
total_time = convert(total_time)
print(f"Finished in: {total_time}")
