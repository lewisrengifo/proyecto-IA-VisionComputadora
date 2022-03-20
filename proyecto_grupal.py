from __future__ import print_function
from statistics import median
import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
import keras
print(f"keras{keras.__version__}")
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from tensorflow.keras.utils import to_categorical

#input data
#auto
img1 = cv2.imread('auto1.jpg',0)
#moto
img2 = cv2.imread('moto2.jpg',0)
print(f'img1 shape: {img1.shape}')
print(f'img2 shape: {img2.shape}')



#edge detection(?
#show
##train tests
'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''



batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 32, 32
##################################################
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#print(f"label xtrain [0] {x_train[0]}")

print(f'image data format: {K.image_data_format()}')
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)



x_train = x_train.astype('float32')
x_test = x_test.astype('float32')#0-255
x_train /= 255
x_test /= 255 # 0-1.0
#print('x_train shape:', x_train.shape)
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
print(f'xtrain: {x_train[0].shape}')
y_train = to_categorical(y_train, num_classes)#[0,0,0,0,0,0,0,0,0,0,1] onehot encoding..
y_test = to_categorical(y_test, num_classes)
###################################################
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))#[0.5,0.2,0.1,0.1,0.0,0.0..]

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model.summary()
'''
model.fit(x_train, y_train,
batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#Guardar el Modelo
model.save('final_model_w12.h5')
'''
