import cv2
import numpy as np
from keras.models import model_from_json
import preprocess as pre
import mapping as mp
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
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split


json_file = open('CNNmodelFinal.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("F:\CODING\ProjectLatex\draft\models\.014-0.783.hdf5")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

img_rows, img_cols = 128, 128

# number of channels
img_channels = 1

path1 = 'F:\CODING\ProjectLatex\dataset\\'  # path of folder of images

listing = os.listdir(path1)
idx = 0
immatrix = []
label = []
for direc in listing:
	path2 = path1 + direc + '\\'
	files = os.listdir(path2)
	for filex in files:
		img = pre.input_image(path2 + filex)
		img_array = np.array(img)
		img_array = np.expand_dims(img_array, axis = 0)
		immatrix.append(img_array)
		label.append(idx)
	idx = idx + 1
	print(idx)

print("DONE STAGE 1")

# %%
data, Label = shuffle(immatrix, label, random_state=2)

# immatrix = np.array(immatrix)
# label = np.array(label)

train_data = [immatrix, label]

# batch_size to train
batch_size = 32
# number of output classes
nb_classes = 95
# number of epochs to train
nb_epoch = 20

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 5

# %%
(X, y) = (train_data[0], train_data[1])

# STEP 1: split X and y into training and testing sets

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=4)

X_train = np.asarray(X_train1, dtype = np.int8)
X_test = np.asarray(X_test1, dtype = np.int8)
y_train = np.asarray(y_train1, dtype = np.int8)
y_test = np.asarray(y_test1, dtype = np.int8)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

loaded_model.evaluate(X_test, Y_test, batch_size = batch_size)