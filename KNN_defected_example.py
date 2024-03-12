
import os

import cv2
from core.classifier import KnnClassifier

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


model = KnnClassifier(general_labels, k=3, splitter_ratio=0.8)
model.load_data(data=data)
model.prepare_model()

print(model.evaluate())
print(model.predict(img_path=data_dir + "/Defected/aug_gauss_noise_cast_def_0_1381.jpeg"))