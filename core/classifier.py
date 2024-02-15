import cv2
import numpy as np
import os

import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import warnings


warnings.filterwarnings('ignore')

""" 
    Classifier Parent Class for classify defected from ok disks
"""
class Classifier(object):
    def __init__(self) -> None:
        self.general_labels = [
            "Defected",
            "OK"
        ]
        self.image_size = 128

""" 
    Nerual Network classifier Implementation
"""
class NNClassifier(Classifier):
    """ Classifier class uses for disk classification as OK or Defected"""
    def __init__(self) -> None:
        super().__init__()
        # default value
        self.model_path = os.path.join("models", "binary_classification_model.keras")
    
    def prepare_model(self, model_path:str="") -> None:
        # loading the model from file
        if(self.model_path == ""):
            self.model_path = model_path
        self.model = tf.keras.models.load_model(self.model_path)
        if(self.model == None):
            raise Exception("invalid model please provide a valid model")
    
    def model_report(self) -> any:
        return self.model.summery()
    
    def predict(self, img_path:str) -> str:
        # read the image in grayscale mode
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = np.array(img)
        img = img.reshape(-1, self.image_size, self.image_size, 1)
        predict = self.model.predict(img)
        return self.general_labels[np.argmax(predict)]


"""
    KNN classifier Implementation
"""
class KnnClassifier(Classifier):
    def __init__(self, k:int=10, splitter_ratio:float=0.8) -> None:
        super().__init__()
        self.image_size = 512
        self.splitter_ration = splitter_ratio
        self.k = k
        self.test_images = []
        self.test_labels = []
        self.train_images = []
        self.train_labels = []
        
    def load_data(self, data_dir:str) -> None:
        images = []
        labels = []
        for label in self.general_labels:
            path = os.path.join(data_dir, label)
            class_num = self.general_labels.index(label)
            for img in os.listdir(path):
                try:
                    f = os.path.join(path, img)
                    img_arr = cv2.imread(f, 0)
                    if (img_arr is None):
                        continue # go for the next image
                    resized_arr = cv2.resize(img_arr, (self.image_size, self.image_size))
                    images.append(resized_arr)
                    labels.append(class_num)
                except Exception as e:
                    print(e)
                    raise e
        images = np.array(images)
        labels = np.array(labels)
        num_samples = len(images)
        num_trains = int(num_samples * self.splitter_ration) 
        self.train_images = images[:num_trains]
        self.train_labels = labels[:num_trains]
        self.test_images = images[num_trains:]
        self.test_labels = labels[num_trains:]
    
    def prepare_model(self, jobs:int=-1):
        x_train = self.train_images
        y_train = self.train_labels
        x_val = self.test_images
        y_val = self.test_labels
        # Normalize the data
        x_train = np.array(x_train) / 255
        x_val = np.array(x_val) / 255
        x_train = x_train.reshape(-1, self.image_size, self.image_size, 1)
        y_train = np.array(y_train)
        x_val = x_val.reshape(-1, self.image_size, self.image_size, 1)
        y_val = np.array(y_val)

        x_train = x_train.reshape((x_train.shape[0], 262144))
        x_val = x_val.reshape((x_val.shape[0], 262144))

        model = KNeighborsClassifier(n_neighbors=self.k, n_jobs=jobs)
        model.fit(x_train, y_train)
        self.model = model
        self.x_val = x_val
        self.y_val = y_val
    
    def model_report(self) -> str|dict:
        return classification_report(self.y_val, self.model.predict(self.x_val), target_names=None)

    def predict(self, img_path:str) -> str:
        test_img = cv2.imread(img_path, 0)
        test_img = cv2.resize(test_img, (self.image_size, self.image_size))
        img = np.array(test_img)
        img = img.reshape((1, test_img.shape[0]*test_img.shape[1]))
        predict = self.model.predict(img)
        return self.general_labels[predict[0]]
