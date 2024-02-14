import numpy as np
import os
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
import tensorflow as tf
from keras.optimizers.legacy import Adam



class Classifier(object):
    """ Classifier class uses for disk classification as OK or Defected"""
    def __init__(self, image_path:str, model_path:str="") -> None:
        self.general_labels = [
            "Defected",
            "OK"
        ]
        self.image_size = 128
        self.model_path = "./binary_classification_model.keras" 
        if(model_path != ""):
            self.model_path = model_path
        if(image_path == ""):
            raise Exception("please insert a image for classification")
        self.image_path = image_path
    
    def prepare_model(self) -> None:
        # loading the model from file
        self.model = tf.keras.models.load_model(self.model_path)
        if(self.model == None):
            raise Exception("invalid model please provide a valid model")
    
    def predict(self) -> str:
        # read the image in grayscale mode
        img = cv2.imread(self.image_path, 0)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = np.array(img)
        img = img.reshape(-1, 128, 128, 1)
        predict = self.model.predict(img)
        return self.general_labels[np.argmax(predict)]



# using the model for predecting a image as defected or Ok
c = Classifier("image.jpeg")
c.prepare_model()
result = c.predict()

print(result)