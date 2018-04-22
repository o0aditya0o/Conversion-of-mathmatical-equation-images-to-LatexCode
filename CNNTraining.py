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
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

import preprocess as pre
# input image dimensions
img_rows, img_cols = 128, 128

# number of channels
img_channels = 1

path1 = '../dataset/'  # path of folder of images

listing = os.listdir(path1)
idx = 0
immatrix = []
label = []
for direc in listing:
	path2 = path1 + direc + '/'
	files = os.listdir(path2)
	for filex in files:
		img = image.load_img(path2+filex, grayscale = True)
		img_array = image.img_to_array(img)
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

'''
img = immatrix[167].reshape(img_rows, img_cols)
plt.imshow(img)
plt.imshow(img, cmap='gray')
print(train_data[0].shape)
print(train_data[1].shape)
'''
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
'''
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

print('y_train shape:', y_train.shape)
print(y_train.shape[0], 'train samples')
print(y_test.shape[0], 'test samples')

print(np.amax(y_test))'''

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
'''
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
'''

checkpoint = ModelCheckpoint(
        filepath='./models/' + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5',
        verbose=1,
        save_best_only=True)

from keras.models import model_from_json
json_file = open('CNNmodel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("CNNmodel.h5")


model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=['accuracy'])

hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                  verbose=1, validation_data=(X_test, Y_test), callbacks = [checkpoint])
'''
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model_json = model.to_json()
with open("CNNmodel2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("CNNmodel2.h5")



from keras.models import model_from_json
json_file = open('CNNmodel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("CNNmodel.h5")

loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

path_img = 'F:\CODING\ProjectLatex\dataset\Sample014\img014-00001.png'
img = pre.input_image(path_img)

img = cv2.resize(img, (128, 128))
img = pre.filter_image(img)
img = pre.otsu_thresh(img)
immatrix = []
img_arr = array(img).flatten()
immatrix.append(img_arr)
inp = np.asarray(immatrix)
inp = inp.reshape(inp.shape[0],128,128,1)
inp = inp.astype('float32')
inp /= 255
print(inp)
    
print(loaded_model.predict_classes(inp))

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
print(Y_test[1:5])'''

'''
	im = pre.input_image(path2 + file)
	im = pre.filter_image(im)
	im = pre.otsu_thresh(im)
	img_arr = array(im).flatten()
	immatrix.append(img_arr)
	label.append(idx)
	'''

