# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 15:36:55 2018

@author: pranay
"""

import cv2
import numpy as np
import imutils
from imutils import contours
from skimage import measure
from keras.preprocessing.image import img_to_array
from keras.models import load_model

model = load_model('ocr_model.h5')
lb = ['0','1','2','3','4','5','6','7','8','9']

real = cv2.imread('roi.jpg',0)
real = imutils.resize(real, width=300)
bordersize=10
real = cv2.copyMakeBorder(real, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255] )
#_, real = cv2.threshold(real, 20, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
image = cv2.imread('roi.jpg',0)
image = imutils.resize(image, width=300)
ret, thresh = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
bordersize=10

thresh = cv2.copyMakeBorder(thresh, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
thresh  = cv2.erode(thresh, kernel, iterations = 1)
labels = measure.label(thresh, neighbors=8, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")
for label in np.unique(labels):
	if label ==0:
		continue
	labelMask = np.zeros(thresh.shape, dtype="uint8")
	labelMask[labels == label] = 255
	numPixels = cv2.countNonZero(labelMask)
	if numPixels > 60:
		mask = cv2.add(mask, labelMask)
		
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = contours.sort_contours(cnts)[0]
final_result = ""
print(len(cnts))
for (i, c) in enumerate(cnts):
	(x,y,w,h) = cv2.boundingRect(c)
	#(x,y,w,h) = (x-10,y-10,x+10,y+10)
	#cv2.rectangle(real, (x-10,y-10),(x+w+10,y+h+10),(0,255,0),2)
	roi = real[y:y+h,x:x+w]
	bordersize=10
	roi = cv2.copyMakeBorder(roi, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255] )
	roi=cv2.resize(roi,(37,50))
	roi=roi.astype('float32')/255.0
	roi=img_to_array(roi)
	roi=np.expand_dims(roi,axis=0)
	prob = model.predict(roi)[0]
	index= np.argmax(prob)
	label=lb[index]
	final_result+=label
print(final_result)
cv2.imshow('real',real)
cv2.imshow('image', image)
cv2.imshow('thresh',thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
