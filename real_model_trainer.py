import os
import re
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf





train_data_dir = os.path.join(os.getcwd(),"")
print(f"train data : {train_data_dir}")

nb_epochs =1

batch_size =None

img_height =None
img_width = None





train_datagen = ImageDataGenerator(rescale=1./255,
    validation_split=0.2) # set validation split


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') # set as validation data

#print(f"labels {label_map}")
#print(f"labels idx {label_map_idx}")

###################
####PUT YOUR CODE HERE ########
model = None



##############3

model.summary()

#readyfortrain=True
readyfortrain=False

#dsv2 = pre_0_otsu

Model_Name= "my_model_name"


AI_BASE_path = os.path.join(os.getcwd(),"AI_models")
model_save_path = os.path.join(AI_BASE_path,Model_Name)




if readyfortrain:

    model.fit_generator(
        train_generator,
        steps_per_epoch = train_generator.samples // batch_size,
        validation_data = validation_generator,
        validation_steps = validation_generator.samples // batch_size,
        epochs = nb_epochs,
        verbose=1)

    print("Saving model.. ")
    model.save(model_save_path)
    print("Model saved")

