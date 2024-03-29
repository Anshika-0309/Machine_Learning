# Implementing simple linear regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()

# print(diabetes.keys())
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])

diabetes_X = diabetes.data[:, np.newaxis, 2]            # using only one feature to keep it understandable(simple linear regression)

diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

diabetes_Y_train = diabetes.target[:-30]
diabetes_Y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_Y_train)
diabetes_Y_predicted = model.predict(diabetes_X_test)

print("Mean squared error is: ", mean_squared_error(diabetes_Y_test, diabetes_Y_predicted))
print("Weights: ", model.coef_)
print("Intercepts: ", model.intercept_)

# Mean squared error is:  3035.060115291269
# Weights:  [941.43097333]
# Intercepts:  153.39713623331644

plt.scatter(diabetes_X_test, diabetes_Y_test)
plt.plot(diabetes_X_test, diabetes_Y_predicted)
plt.show()                                             # show the simple linear regression plot along with the scattered data
