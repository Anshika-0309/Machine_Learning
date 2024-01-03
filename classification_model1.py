# Importing required modules
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# Loading datasets
iris = datasets.load_iris()

# Printing description and features
# print(iris.DESCR)

features = iris.data
labels = iris.target

# Training the classifier
clf = KNeighborsClassifier()
clf.fit(features, labels)

# Predicting labels after training
predicted = clf.predict([[1.2, 4.5, 6.4, 2.3]])
print("output -> ", predicted)
# output -> [2]
