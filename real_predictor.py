#! /usr/bin/env python3
# ! -*- coding: utf-8 -*-
import gc
import glob
from unicodedata import category
from keras.models import load_model
import os
import cv2
import numpy as np
import urllib.request
from matplotlib import pyplot as plt
import json
# confusion matrix in sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd




Model_Name= "my_model"
model_save_path = "final_model_w12.h5"
model = load_model(model_save_path)


img_np = cv2.imread("dog2.jpg")  # cv2.IMREAD_COLOR in OpenCV 3.1
#resize for print
scale_percent = 130 # percent of original size
width = int(img_np.shape[1] * scale_percent / 100)
height = int(img_np.shape[0] * scale_percent / 100)
dim_pre = (width, height)
img1Resized = cv2.resize(img_np,dim_pre, interpolation= cv2.INTER_AREA)

dim = (32, 32)
resized= cv2.resize(img_np, dim)
print(f'shape: {img_np.shape}')
img_to_process = np.expand_dims(resized, axis=0)

pred = model.predict(img_to_process)
print(f'pred: {pred}')
top = np.argmax(pred)
print(f'top: {top}')
category = ''
if top == 0 :
    category = 'airplane'
elif top ==1:
    category = 'automobile'
elif top == 2:
    category = 'bird'
elif top == 3:
    category = 'cat'
elif top ==4:
    category = 'deer'
elif top ==5:
    category = 'dog'
elif top ==6:
    category = 'frog'
elif top ==7:
    category='horse'
elif top ==8:
    category = 'ship'
else :
    category= 'truck'

    
print(f'La categoria de la imagen es: {category}')
category2 = 'LA CATEGORIA ES: ' + category
cv2.imshow(category2, img_np)
cv2.waitKey(0)

