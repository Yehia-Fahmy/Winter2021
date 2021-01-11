import numpy as np
import os
import cv2
import time as t
import random
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sys import getsizeof


def load_data():
    print("Loading data...")
    train_data = []     # list to hold the training data
    test_data = []      # list to hold the testing data
    err = 0             # variable to keep track of any missed images
    for catagory in CATAGORIES:  # for each catagory
        counter = 0  # counter to add every 10th element to testing
        path = os.path.join(DATADIR, catagory)
        classification = CATAGORIES.index(catagory)
        for img in os.listdir(path):  # for each image
            counter = counter + 1  # index the counter
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # read the image
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # assert the correct size
                if counter % 10 == 0:
                    test_data.append([img_array, classification])  # adds the data to the testing list
                else:
                    train_data.append([img_array, classification])  # adds the data to the training list
            except Exception as e:
                err = err + 1  # counts the errors
    return train_data, test_data


def crop_data(data):
    print("Cropping data...")
    err = 0             # variable to keep track of any missed images
    new_data = []       # list to hold the data
    try:
        for img in data:                # for each image
            for i in range(round(IMG_SIZE / NEW_SIZE)):          # going through the rows
                for k in range(round(IMG_SIZE / NEW_SIZE)):      # going through the columns
                    new_data.append([img[0][i * NEW_SIZE:(i + 1) * NEW_SIZE, k * NEW_SIZE:(k + 1) * NEW_SIZE],
                                     img[1]])  # adds the data as a list
    except Exception as e:
        err = err + 1
    return new_data


def flip_data(data):
    print("Flipping data...")
    new_data = []       # list to hold the data
    err = 0             # variable to keep track of any missed images
    try:
        for img in data:
            new_data.append([img[0], img[1]])  # adds the original image
            new_data.append([np.flip(img[0], axis=0), img[1]])  # adds the image flipped horizontally
            new_data.append([np.flip(img[0], axis=1), img[1]])  # adds the image flipped vertically
            new_data.append([np.flip(np.flip(img[0], axis=1), axis=0), img[1]])  # adds the image flipped both ways
    except Exception as e:
        err = err + 1
    return new_data


def rotate_data(data):
    print("Rotating data...")
    new_data = []  # list to hold the data
    err = 0  # variable to keep track of any missed images
    try:
        for img in data:
            new_data.append(img)  # adds the original
            new_data.append([cv2.rotate(img[0], cv2.ROTATE_90_CLOCKWISE), img[1]])
            new_data.append([cv2.rotate(img[0], cv2.ROTATE_90_COUNTERCLOCKWISE), img[1]])
            new_data.append([cv2.rotate(img[0], cv2.ROTATE_180), img[1]])  # adds the other 3 orientations
    except Exception as e:
        err = err + 1
    return new_data


def shuffle_data(data):
    print("Shuffling data...")
    random.shuffle(data)        # randomly shuffling data
    return data


def split_data(data):
    print("Splitting data...")
    X = []  # list of images
    y = []  # list of labels

    for features, label in data:  # splits the data
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, NEW_SIZE, NEW_SIZE, 1)
    return X,y


def normalize_data(X_train, y_train, X_test, y_test):
    print("Normalizing data...")
    X_train = np.array(X_train)
    y_train = np.array(y_train)         # ensuring the lists are numpy arrays
    y_train = to_categorical(y_train)
    X_train = X_train / 255             # normalizing the data

    X_test = np.array(X_test)
    y_test = np.array(y_test)           # ensuring the lists are numpy arrays
    y_test = to_categorical(y_test)
    X_test = X_test / 255               # normalizing the data
    return X_train, y_train, X_test, y_test


def print_sizes(data):
    for item in data:
        size = getsizeof(item) / 1e9
        print(f'Size of {data.index(item)}: {size} GB')


def save_data(X_train, y_train, X_test, y_test):
    print(f"Saving X_test data...")
    pickle_out = open("X_test.pickle", "wb")
    pickle.dump(X_test, pickle_out)
    pickle_out.close()

    print(f"Saving y_train data...")
    pickle_out = open("y_train.pickle", "wb")
    pickle.dump(y_train, pickle_out)
    pickle_out.close()

    print(f"Saving y_test data...")
    pickle_out = open("y_test.pickle", "wb")
    pickle.dump(y_test, pickle_out)
    pickle_out.close()

    print(f"Saving X_train data...")
    pickle_out = open("X_train.pickle", "wb")
    pickle.dump(X_train, pickle_out)
    pickle_out.close()
    dim = round(X_train.shape[0]/3)
    for i in range(3):
        print(f'Saving X_train({i}) data...')
        pickle_out = open(f"X_train({i}).pickle", "wb")
        pickle.dump(X_train[(i * dim):((i+1) * dim)], pickle_out)
        pickle_out.close()


def print_img(data, index):
    print(f"classification = {data[index][1]}")
    plt.imshow(data[index][0], cmap='gray')
    plt.show()


DATADIR = "/Users/Yehia/Desktop/Enhanced_Pictures(7)"  # directory of all the pictures
CATAGORIES = ["One", "Two", "Three", "Four", "Five", "Six", "Seven"]
IMG_SIZE = 480  # size of the images we will import
NEW_SIZE = 120  # size of images after resizing

start_time = t.time()
print("Starting...")

train_data, test_data = load_data()
print(f"{len(train_data)} training examples, {len(test_data)} testing examples")

train_data = crop_data(train_data)
test_data = crop_data(test_data)
print(f"{len(train_data)} training examples, {len(test_data)} testing examples")

train_data = flip_data(train_data)
test_data = flip_data(test_data)
print(f"{len(train_data)} training examples, {len(test_data)} testing examples")

train_data = rotate_data(train_data)
test_data = rotate_data(test_data)
print(f"{len(train_data)} training examples, {len(test_data)} testing examples")

train_data = shuffle_data(train_data)
test_data = shuffle_data(test_data)

X_train, y_train = split_data(train_data)
X_test, y_test = split_data(test_data)

# X_train, y_train, X_test, y_test = normalize_data(X_train, y_train, X_test, y_test)
print_sizes([X_train, y_train, X_test, y_test])
save_data(X_train, y_train, X_test, y_test)

total_time = t.time() - start_time
total_time = round(total_time, 2)
print(f"{len(train_data)} training examples, {len(test_data)} testing examples")
print("Finished in: ", total_time, "s!")
