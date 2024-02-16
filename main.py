
from core.classifier import KnnClassifier, NNClassifier


# using the model for predecting a image as defected or Ok
img_path = "path/to/image.jpg"
data_dir = "path/to/dataset_dir"
c = KnnClassifier(k=5)
c.load_data(data_dir)
c.prepare_model()
clas = c.predict(img_path)
print("knn evaluation: ", c.evaluate())
print("the result of KNN is : ", clas)

c2 = NNClassifier()
c2.load_data(data_dir)
c2.prepare_model()
pre = c2.predict(img_path)
print("nn evaluation: ", c2.evaluate())
print("the result of NN is : ", pre)