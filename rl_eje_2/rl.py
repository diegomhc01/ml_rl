from sklearn import linear_model
import pandas as pd
#!pip install statsmodels
import matplotlib.pyplot as plt
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("california_housing_train.csv")
test = pd.read_csv('california_housing_test.csv')
# Eliminamos los NA de los datos, que a sklearn no le gustan
train.dropna(inplace=True)
test.dropna(inplace=True)

#print(train.describe())


X = train.drop(columns='median_house_value')
Y =  train['median_house_value']
print(X)
print(Y)
X_train = np.array(X)
lm = linear_model.LinearRegression()
lm.fit(X, Y)
#print('El R^2 de nuestro modelo para los datos de entrenamiento es de', lm.score(X, Y))


Xtest = test.drop(columns='median_house_value')
Ytest = test['median_house_value']
lm = linear_model.LinearRegression()
lm.fit(Xtest, Ytest)
#print('El R^2 de nuestro modelo para los datos de entrenamiento es de', lm.score(Xtest, Ytest))

#z_pred = lm.predict(X_train)
#datos = pd.DataFrame(z_pred)
#datos.to_csv('z_pred.csv')
#print(z_pred)

pickle.dump(lm, open('modelo.p', 'wb'))