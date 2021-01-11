import keras

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

import numpy as np



def build_network():
    model = Sequential()
    # 1st layer
    model.add(Conv2D(filters=96, input_shape=IMG_SHAPE, kernel_size=(11, 11), strides=(4, 4), padding='valid'))
    model.add(Activation('relu'))
    # pooling
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # 2nd layer
    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # pooling
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # 3rd layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))

    '''# 4th layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))

    # 5th layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(2, 2), padding='valid'))

    # 6th layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))

    # 7th layer
    model.add(Dense(4096, activation='relu'))

    # 8th layer
    model.add(Dense(1000, activation='softmax'))'''

    model.summary()

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


IMG_SHAPE = (227, 227, 3)
np.random.seed(1000)
print("Starting...")
build_network()
print(f"Done!")
