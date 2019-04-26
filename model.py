import h5py
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import load_model
from keras import backend as K
import numpy as np
from os import listdir
from os.path import isfile, join
import csv
from sklearn.metrics import f1_score

CNN_MODEL_PATH="models/cnn/model_512.h5"
GLCM_MODEL_PATH=""
def classify_cnn_model(model_path):
    if K.image_data_format() == 'channels_first':
        input_shape = (3, 150, 150)
    else:
        input_shape = (150, 150, 3)

    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.load_weights(model_path)
    return model

def predict(task,image_path):
    if task=="cnn":
        model_path=CNN_MODEL_PATH
        model=classify_cnn_model(model_path)
        image = load_img(image_path, target_size=(150, 150))
        img = img_to_array(image)
        img = np.expand_dims(img, axis=0)
        preds = model.predict_classes(img)
        if preds[0][0]==0 or preds[0][0]=="0":
            return "No Pneumonia Present"
        else:
            return "Pneumonia Present"


    elif task=="glcm":
        model_path=GLCM_MODEL_PATH
        return