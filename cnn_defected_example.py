
import os
import cv2

from core.classifier import CNNClassifier

img_path = "/Users/amir/Desktop/computer_vision/project/DataSet1/Defected/aug_gauss_noise_cast_def_0_2148.jpeg"

general_labels = [
    "Defected",
    "OK"
]
data_dir = "/Users/amir/Desktop/computer_vision/project/DataSet1"
images = []
labels = []
for label in general_labels:
    path = os.path.join(data_dir, label)
    class_num = general_labels.index(label)
    for img in os.listdir(path):
        try:
            f = os.path.join(path, img)
            img_arr = cv2.imread(f, 0)
            if (img_arr is None):
                continue # go for the next image
            resized_arr = cv2.resize(img_arr, (128, 128))
            images.append(resized_arr)
            labels.append(class_num)
        except Exception as e:
            print(e)
            raise e

data = {
    'x': images,
    'y': labels
}


model = CNNClassifier(labels=general_labels, name="binary_classification_model")
model.load_data(data=data)
model.prepare_model()
print(model.evaluate())
print(model.predict(img_path=img_path))
