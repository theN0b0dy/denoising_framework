import os
import cv2

from core.classifier import NNClassifier

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
c2 = NNClassifier(labels=general_labels, name="binary_classification_model", image_size=128)
c2.load_data(data=data)
c2.prepare_model()
c2.train(learning_rate=0.001, epochs=50)
c2.plot_history()
print("nn evaluation: ", c2.evaluate())