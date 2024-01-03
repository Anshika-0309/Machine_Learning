# Logistic regression classifier to classify whether an iris is verginica (with value 2) or not

# Importing required modules
from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Loading datasets
iris = datasets.load_iris()
# print(list(iris.keys()))
# print(iris.data)
# print(iris.target)

x = iris.data[:, 3:]                                 # taking only one feature into consideration for simplicity
y = (iris.target == 2).astype(np.int_)               # if type is verginica (i.e. 2), return 1 else 0
# print(x)
# print(y)

# Training the logistic regression classifier
clf = LogisticRegression()
clf.fit(x, y)
example = clf.predict([[2.4]])
# print(example)                                      # prints [1]

# Using matplotlib to plot the visualization
x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_prob = clf.predict_proba(x_new)
# print(y_prob)
plt.plot(x_new, y_prob[:, 1])
plt.show()
