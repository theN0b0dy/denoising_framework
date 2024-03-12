import pandas as pd
import cv2
import os

from core.classifier import NNClassifier

# read the labels data
df = pd.read_excel('labels.xlsx')
general_labels = [
    "Gaussian",
    "Periodic", 
    "Salt"
]
model = NNClassifier(labels=general_labels, splitter_ration=0.7, name='noise-type-detection')

base_dir = "path/to/dataset"
images = []
labels = []
for index, row in df.iterrows():
    f_name = os.path.join(base_dir, row['Noisy Image'])
    # gray scale mode
    img = cv2.imread(f_name, 0)
    img = cv2.resize(img, (128, 128))
    images.append(img)
    l = row['Noise Type']
    if (l == "Gaussian"):
        labels.append(0)
    elif (l == "Periodic"):
        labels.append(1)
    elif (l == "Salt"):
        labels.append(2)

data = {
    'x': images,
    'y': labels
}
model.load_data(data=data)
model.prepare_model()
model.train(learning_rate=0.000001, epochs=30)
print(model.evaluate())