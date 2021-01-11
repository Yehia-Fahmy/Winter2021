import numpy as np
import os
import cv2
import time as t
import random
import pickle
import matplotlib.pyplot as plt

start_time = t.time()
print("Starting...")
# DATADIR = "\Users\Yehia\Desktop\Enhanced_Pictures(2)"
# DATADIR = "C:/Users/Yehia/Desktop/Enhanced_Pictures(2)"
DATADIR = input("Enter path to pictures:")                  # inputs the path to the pictures
CATAGORIES = ["Hydrophobic", "Hydrophilic"]                 # list of different classifications
full_size_data = []         # some intermediate variables
cropped_data = []
flipped_data = []
training_data = []          # where the data will finally be stored
IMG_SIZE = 480              # size of the images we will import
NEW_SIZE = 160              # size of images after resizing


def load_data():
    print("Loading data...")
    err = 0         # variable to keep track of any missed images
    for catagory in CATAGORIES:             # for every catagory
        path = os.path.join(DATADIR, catagory)
        class_num = CATAGORIES.index(catagory)
        for img in os.listdir(path):        # for every image
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)   # reads the image
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))     # confirms it is the correct size
                full_size_data.append([img_array, class_num])   # adds the data as a list
            except Exception as e:
                err = err + 1       # counts the errors we have
        print(len(full_size_data), "training examples (", err, "errors )")


def crop_data(data):
    print("Cropping data...")
    for img in data:        # for each image
        for i in range(3):      # going through the rows
            for k in range(3):      # going through the columns
                cropped_data.append([img[0][i*NEW_SIZE:(i+1)*NEW_SIZE, k*NEW_SIZE:(k+1)*NEW_SIZE],
                                     img[1]])       # adds the data as a list
    print(len(cropped_data), "training examples")


def flip_data(data):
    print("Flipping data...")
    for img in data:        # for each image
        flipped_data.append([img[0], img[1]])   # adds the original image
        flipped_data.append([np.flip(img[0], axis=0), img[1]])      # adds the image flipped horizontally
        flipped_data.append([np.flip(img[0], axis=1), img[1]])      # adds the image flipped vertically
        flipped_data.append([np.flip(np.flip(img[0], axis=1), axis=0), img[1]]) # adds the image flipped both ways
    print(len(flipped_data), "training examples")


def rotate_data(data):
    print("Rotating data...")
    for img in data:        # for each image
        training_data.append(img)   # adds the original
        training_data.append([cv2.rotate(img[0], cv2.ROTATE_90_CLOCKWISE), img[1]])
        training_data.append([cv2.rotate(img[0], cv2.ROTATE_90_COUNTERCLOCKWISE), img[1]])
        training_data.append([cv2.rotate(img[0], cv2.ROTATE_180), img[1]])      # adds the other 3 orientations
    print(len(training_data), "training examples")


def shuffle_data(data):
    print("Shuffling data...")
    random.shuffle(data)        # randomly shuffles the data
    return data


def split_data(data):
    print("Splitting data...")
    X = []      # list of images
    y = []      # list of labels
    for features, label in data:        # splits the data
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, 160, 160, 1)
    return X, y


def save_data(X, y):
    print("Saving...")
    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


def show(img):
    plt.imshow(img, cmap='gray')
    plt.show()


load_data()

crop_data(full_size_data)

flip_data(cropped_data)

rotate_data(flipped_data)

shuffle_data(training_data)

X_data, y_data = split_data(training_data)

save_data(X_data, y_data)

total_time = t.time() - start_time

total_time = round(total_time, 2)

print("Finished in: ", total_time, "s!")