# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 12:03:19 2018

@author: pranay
"""

import keras
import cv2
import os
import numpy as np

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

# Saving and loading model and weights
from keras.models import model_from_json
from keras.models import load_model

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

people = [dI for dI in os.listdir('dataSet') if os.path.isdir(os.path.join('dataSet',dI))]
num_classes = len(people)
img_data_list = []
labels = []
valid_images = [".jpg",".gif",".png"]

for index, person in enumerate(people):
  print(index)
  dir_path = 'dataSet/' + person
  for img_path in os.listdir(dir_path):
    name, ext = os.path.splitext(img_path)
    if ext.lower() not in valid_images:
        continue

    img_data = cv2.imread(dir_path + '/' + img_path)
    # convert image to gray
    img_data=cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    img_data = cv2.resize(img_data, (37, 50))
    img_data_list.append(img_data)
    labels.append(index)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')

labels = np.array(labels ,dtype='int64')
# scale down(so easy to work with)
img_data /= 255.0
img_data= np.expand_dims(img_data, axis=4)

# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)
# print(Y)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=5)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Defining the model
input_shape=img_data[0].shape
print(input_shape)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

model.save('ocr_model.h5')

