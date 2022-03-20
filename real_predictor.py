#! /usr/bin/env python3
# ! -*- coding: utf-8 -*-
import gc
import glob
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
model_save_path = "path_to_my_model.h5"
model = load_model(model_save_path)


img_np = cv2.imread("cafe.jpg")  # cv2.IMREAD_COLOR in OpenCV 3.1

dim = (32, 32)
resized= cv2.resize(img_np, dim)
print(f'shape: {img_np.shape}')
img_to_process = np.expand_dims(resized, axis=0)

#img_RGB = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
# cv2.imshow("imgs",img_RGB)
# cv2.waitKey(0)
#img_ar = np.asarray(img_np)
#img_ar = img_ar / 255
# print(f"shape : {img_ar.shape}")
#model_input = np.expand_dims(img_ar, axis=0)
pred = model.predict(img_to_process)
print(f'pred: {pred}')
top = np.argmax(pred)
print(f'top: {top}')
