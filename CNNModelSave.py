#KERAS  root_l@B
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.preprocessing import image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import array

nb_classes = 95
nb_pool = 2
img_rows = 128
img_cols = 128

model = Sequential()

model.add(Convolution2D(32 , 5, 5,
                        border_mode='valid',
                        input_shape=(img_rows, img_cols, 1)))
convout1 = Activation('relu')
model.add(convout1)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Convolution2D(64 , 5, 5))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(84))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model_json = model.to_json()
with open("CNNmodelFinal.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5

