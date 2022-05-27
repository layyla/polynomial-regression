import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures

#importing data and dropping null
data = pd.read_csv('dataset_cars.csv')
data.drop(data.loc[data['kilometers'] == 0].index, inplace=True)

#encoding text data
encode = LabelEncoder()
encode.fit(list(data['car_maker'].values))
data['car_maker'] = encode.transform(list(data['car_maker'].values))

X = data.iloc[:, 0:3]
Y = data['price']

#feature scaling
X = np.array(X)
Normalized_X = np.zeros((X.shape[0], X.shape[1]))
for i in range(X.shape[1]):
    Normalized_X[:, i] = (X[:, i] - min(X[:, i]))/(max(X[:, i])-min(X[:, i]))
X = pd.DataFrame(Normalized_X, columns=(data.keys())[0:3])

#feature selection using correlation coefficient score
corr = data.corr()
top_feature = corr.index[abs(corr['price']) > 0.05]
top_feature = top_feature.delete(-1)
X = X[top_feature]

#Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

#Polynomial regression
polynom = PolynomialFeatures(degree=5)
poly_model = linear_model.LinearRegression()

# transforming X of training set to be polynomial
X_train_poly = polynom.fit_transform(X_train)

#fitting data in model
poly_model.fit(X_train_poly, y_train)

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)

# predicting on test data-set
prediction = poly_model.predict(polynom.fit_transform(X_test))

print('Mean Square Error of training =', metrics.mean_squared_error(y_train, y_train_predicted))
print('Mean Square Error of test =', metrics.mean_squared_error(y_test, prediction))
