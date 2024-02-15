
from core.classifier import KnnClassifier, NNClassifier


# using the model for predecting a image as defected or Ok
img_path = "img.jpeg"
c = KnnClassifier(k=5)
c.load_data("path/to/dataset")
c.prepare_model()
clas = c.predict(img_path)
print("the result of KNN is : ", clas)

c2 = NNClassifier()
c2.prepare_model()
pre = c2.predict(img_path)
print("the result of NN is : ", pre)