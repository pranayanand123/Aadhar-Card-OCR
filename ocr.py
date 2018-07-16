# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 11:30:32 2018

@author: pranay
"""

import cv2
import numpy as np
import imutils
from PIL import Image
import matplotlib.pyplot as plt

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
image  = cv2.imread('aadhar_card3.jpg')
image = imutils.resize(image, width = 300)
gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray,5)
ret3, thresh3 = cv2.threshold(gray,100,255, cv2.THRESH_BINARY)
ret, thresh = cv2.threshold(gray,100,255, cv2.THRESH_BINARY_INV)
tophat = cv2.morphologyEx(thresh, cv2.MORPH_TOPHAT, rectKernel)
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,
	ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh2 = cv2.threshold(gradX, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, sqKernel)
cnts = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
locs = []
for (i, c) in enumerate(cnts):
	# compute the bounding box of the contour, then use the
	# bounding box coordinates to derive the aspect ratio
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)
 
	# since credit cards used a fixed size fonts with 4 groups
	# of 4 digits, we can prune potential contours based on the
	# aspect ratio
	if ar > 6 and ar < 11.5:
		#cv2.rectangle(image, (x,y),(x+w,y+h),(255,0,0),1)
		#locs.append((x, y, w, h))
		# contours can further be pruned on minimum/maximum width
		# and height
		if (w > 100 and w < 111) and (h > 11 and h < 19):
			# append the bounding box region of the digits group
			# to our locations list
			locs.append((x, y, w, h))
print(locs)
locs = locs[0]
print(locs)
cv2.rectangle(image, (locs[0], locs[1]),(locs[0]+locs[2],locs[1]+locs[3]) , (255,0,0), 1)
roi = thresh3[locs[1]:locs[1]+locs[3],locs[0]:locs[0]+locs[2]]
cv2.imwrite('roi.jpg',roi)
#cv2.drawContours(image, cnts,-1, (0,0,255), 1)
plt.imshow(image)
cv2.imshow('roi', roi)
cv2.imshow('thresh2', thresh2)
cv2.imshow('gradX', gradX)
cv2.imshow('thresh', thresh)
cv2.imshow('tophat', tophat)
cv2.imshow('gray', gray)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
