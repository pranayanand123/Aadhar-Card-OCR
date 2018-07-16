# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 12:41:04 2018

@author: pranay
"""

from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
lb = ['0','1','2','3','4','5','6','7','8','9']

model = load_model('ocr_model.h5')

image=cv2.imread('roi9.jpg')
print(image.shape)
image=cv2.resize(image,(37,50))
image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
print(image.shape)
image=image.astype('float32')/255.0
image=img_to_array(image)
image=np.expand_dims(image,axis=0)

prob = model.predict(image)[0]
index= np.argmax(prob)
label=lb[index]
print(label)
print(prob[index]*100)
