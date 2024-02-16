import cv2
import numpy as np
import os

import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from keras.optimizers.legacy import Adam

import matplotlib.pyplot as plt

import warnings


warnings.filterwarnings('ignore')

""" 
    Classifier Parent Class for classify defected from ok disks
"""
class Classifier(object):
    def __init__(self) -> None:
        self.image_size = 128
        self.splitter_ration = 0.8
        self.train_images = []
        self.train_labels = []

    def load_data(self, data:dict) -> None:
        images = np.array(data['x'])
        labels = np.array(data['y'])
        if(len(images) == 0 or len(labels) == 0):
            raise Exception("empty database please provide a valid dataset")
        num_samples = len(images)
        num_trains = int(num_samples * self.splitter_ration) 
        self.train_images = images[:num_trains]
        self.train_labels = labels[:num_trains]
        self.test_images = images[num_trains:]
        self.test_labels = labels[num_trains:]
    
    def prepare_model(self) -> None:
        x_train = self.train_images
        y_train = self.train_labels
        x_val = self.test_images
        y_val = self.test_labels
        # Normalize the data
        x_train = np.array(x_train) / 255
        x_val = np.array(x_val) / 255

        self.x_train = x_train.reshape(-1, self.image_size, self.image_size, 1)
        self.y_train = np.array(y_train)
        self.x_val = x_val.reshape(-1, self.image_size, self.image_size, 1)
        self.y_val = np.array(y_val)
    
    def train(self) -> None:
        pass

""" 
    Nerual Network classifier Implementation
"""
class NNClassifier(Classifier):
    """ Classifier class uses for disk classification as OK or Defected"""
    def __init__(self, labels:list[str], splitter_ration:float=0.8,name:str="sample", image_size:int=128) -> None:
        super().__init__()
        # default value
        self.splitter_ration = splitter_ration
        self.general_labels = labels
        self.name = name
        self.image_size = image_size
        self.history = None
    
    def prepare_model(self, model_path:str="") -> None:
        if(len(self.train_images) != 0): # if we want to train the models
            super().prepare_model()
        # otherwise 
        # loading the model from file
        self.model_path = os.path.join("models", f"{self.name}.keras")
        if(model_path != ""):
            self.model_path = model_path
        try:
            self.model = tf.keras.models.load_model(self.model_path)
        except OSError:
            return
        if(self.model == None):
            raise Exception("invalid model please provide a valid model or train new model with .train() function")

    def load_data(self, data:dict) -> None:
        super().load_data(data)
    
    def evaluate(self) -> any:
        return self.model.evaluate(self.x_val, self.y_val, batch_size=128)
    
    def predict(self, img_path:str) -> str:
        # read the image in grayscale mode
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = np.array(img)
        img = img.reshape(-1, self.image_size, self.image_size, 1)
        predict = self.model.predict(img)
        return self.general_labels[np.argmax(predict)]
    
    def plot_history(self) -> None:
        if self.history is None:
            raise Exception("first you need to train() your model then you can plot the history of learing")
        # summarize history for accuracy
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def train(self, learning_rate:float=0.001, epochs:int=20) -> None:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.2, # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip = True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        datagen.fit(self.x_train)
        # simple network for multiclass classification
        self.model = Sequential()
        self.model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(self.image_size,self.image_size,1)))
        self.model.add(MaxPool2D())
        self.model.add(Flatten())
        self.model.add(Dense(self.image_size,activation="relu"))
        self.model.add(Dense(len(self.general_labels), activation="softmax"))

        opt = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=opt ,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])

        self.history = self.model.fit(self.x_train,self.y_train,epochs=epochs, validation_data = (self.x_val, self.y_val), batch_size=128)
        # saving the model
        self.model.save(self.model_path)
        


"""
    KNN classifier Implementation
"""
class KnnClassifier(Classifier):
    def __init__(self, labels:list[str], k:int=1, splitter_ratio:float=0.8) -> None:
        super().__init__()
        self.general_labels = labels
        self.image_size = 512
        self.splitter_ration = splitter_ratio
        self.k = k
        self.test_images = []
        self.test_labels = []
        self.train_images = []
        self.train_labels = []
        
    def load_data(self, data:dict) -> None:
        super().load_data(data)
    
    def prepare_model(self, jobs:int=-1):
        super().prepare_model()
        self.x_train = self.x_train.reshape((self.x_train.shape[0], 262144))
        self.x_val = self.x_val.reshape((self.x_val.shape[0], 262144))

        model = KNeighborsClassifier(n_neighbors=self.k, n_jobs=jobs)
        model.fit(self.x_train, self.y_train)
        # set the model
        self.model = model
    
    def evaluate(self) -> str|dict:
        return classification_report(self.y_val, self.model.predict(self.x_val), target_names=None)

    def predict(self, img_path:str) -> str:
        test_img = cv2.imread(img_path, 0)
        test_img = cv2.resize(test_img, (self.image_size, self.image_size))
        img = np.array(test_img)
        img = img.reshape((1, test_img.shape[0]*test_img.shape[1]))
        predict = self.model.predict(img)
        return self.general_labels[predict[0]]
