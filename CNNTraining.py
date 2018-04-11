#KERAS
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import array
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

import preprocess as pre
# input image dimensions
img_rows, img_cols = 128, 128

# number of channels
img_channels = 1

path1 = 'F:\CODING\ProjectLatex\dataset\\'  # path of folder of images

listing = os.listdir(path1)
idx = 1
immatrix = []
label = []
for direc in listing:
    path2 = path1 + direc + '\\'
    files = os.listdir(path2)
    for file in files:
        im = pre.input_image(path2 + file)
        im = pre.filter_image(im)
        im = pre.otsu_thresh(im)
        img_arr = array(im).flatten()
        immatrix.append(img_arr)
        label.append(idx)
    idx = idx + 1
    print(idx)
'''
f = open('images_array.txt', 'w')
f.write(immatrix)
f.close()

f = open('labels_array.txt', 'w')
f.write(label)
f.close()
'''
print("DONE STAGE 1")



# %%
data, Label = shuffle(immatrix, label, random_state=2)
train_data = [data, Label]

img = immatrix[167].reshape(img_rows, img_cols)
plt.imshow(img)
plt.imshow(img, cmap='gray')
print(train_data[0].shape)
print(train_data[1].shape)

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

i = 100
plt.imshow(X_train[i, 0], interpolation='nearest')
print("label : ", Y_train[i,:])

model = Sequential()

model.add(Convolution2D(6 , 5, 5,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
convout1 = Activation('relu')
model.add(convout1)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Convolution2D(16, 5, 5))
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
model.compile(loss='categorical_crossentropy', optimizer='adadelta')

hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                 show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))

with open('CNNClassifier.pkl', 'wb') as f:
    pickle.dump(hist, f)


train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(nb_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
# print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])




#%%

score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(model.predict_classes(X_test[1:5]))
print(Y_test[1:5])