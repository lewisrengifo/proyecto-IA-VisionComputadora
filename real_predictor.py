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
model_save_path = "model path"
model = load_model(model_save_path)

img_np = cv2.imread("image_path.jpg")  # cv2.IMREAD_COLOR in OpenCV 3.1
img_RGB = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
# cv2.imshow("imgs",img_RGB)
# cv2.waitKey(0)
img_ar = np.asarray(img_RGB)
img_ar = img_ar / 255
# print(f"shape : {img_ar.shape}")
model_input = np.expand_dims(img_ar, axis=0)
pred = model.predict(model_input)

top = np.argmax(pred)
