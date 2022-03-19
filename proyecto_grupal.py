from statistics import median
import cv2
from matplotlib import pyplot as plt
import os
import numpy as np

#input data
#auto
img1 = cv2.imread('auto1.jpg',0)
#moto
img2 = cv2.imread('moto2.jpg',0)
print(f'img1 shape: {img1.shape}')
print(f'img2 shape: {img2.shape}')


#preprocesamiento
#resize 1
scale_percent = 30 # percent of original size
width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)
dim = (width, height)
img1Resized = cv2.resize(img1,dim, interpolation= cv2.INTER_AREA)
print(f'shape: {img1Resized.shape}')
#resize 2
scale_percent = 30 # percent of original size
width = int(img2.shape[1] * scale_percent / 100)
height = int(img2.shape[0] * scale_percent / 100)
dim = (width, height)
img2Resized = cv2.resize(img2,dim, interpolation= cv2.INTER_AREA)
print(f'shape: {img2Resized.shape}')
#resize show
cv2.imshow('auto', img1Resized)
cv2.imshow('moto', img2Resized)
cv2.waitKey(0)
#construcion de un histograma para tener criterios de seleccion de filtros
plt.hist(img1Resized.ravel(),256,[0,256])
plt.hist(img2Resized.ravel(),256,[0,256])
plt.show()

#filtros
image_tofilter = img1Resized
normal_blur = cv2.blur(image_tofilter, (5,5))
median_blur = cv2.medianBlur(image_tofilter, 5)
gaussian_blur = cv2.GaussianBlur(image_tofilter,(5,5),0)
cv2.imshow('normal blur',normal_blur)
cv2.imshow('median blur',median_blur)
cv2.imshow('gaussian blur',gaussian_blur)
cv2.waitKey(0)
#edge detection(?
#show

