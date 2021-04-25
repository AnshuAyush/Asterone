import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets , linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()


# ['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename']
diabetes_x = diabetes.data[: , np.newaxis , 2]
print(diabetes_x)
diabetes_x_train = diabetes_x[: -30]
diabetes_x_test = diabetes_x[-20 : ]
diabetes_y_train = diabetes.target[:  -30]
diabetes_y_test = diabetes.target[-20 : ]
model = linear_model.LinearRegression()
model.fit(diabetes_x_train , diabetes_y_train)
diabities_y_predict = model.predict(diabetes_x_test)
print("Error is " , mean_squared_error(y_true = diabetes_y_test , y_pred = diabities_y_predict))
print("weights" , model.coef_)
print("Intercept" , model.intercept_)
plt.scatter(diabetes_x_test , diabetes_y_test)
plt.plot(diabetes_x_test , diabities_y_predict)
plt.show()


# slicing 