import numpy as np
import pandas as pd
import keras

#definding model libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import math

from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import confusion_matrix

import random # shuffling the class
data = pd.read_csv("data.csv")

data = data.sample(frac=1)
print(data)

y = data.iloc[:,0].values  # numpy array
X = data.iloc[:,1:].values#features
print X
X = X*math.pow(10,11)
print X


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

random.shuffle(X)

classifier = Sequential()
classifier.add(Dense(activation = 'relu', input_dim = 7, units = 7,
                     kernel_initializer = 'normal'))
# classifier.add(Dropout(0.9))
classifier.add(Dense(activation = 'sigmoid', units = 1,
                     kernel_initializer = 'uniform'))
# classifier.add(Dropout(0.9))

classifier.compile(  optimizer = 'adam', loss='mean_squared_error', metrics = ['accuracy'])
classifier.fit(X, y, epochs = 10)

pred_y = classifier.predict_classes(X_test) #predicting labels
c_table=confusion_matrix(y_test,pred_y)


print('Confusion Matrix : \n', c_table )

#



#
#
# from sklearn import neighbors, linear_model
# knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', p=1)
#
#
# knn_model_1 = knn.fit(X_train, y_train)
# print('k_NN accuracy for test set: %f' % knn_model_1.score(X, y))
#
# from sklearn.metrics import classification_report
# y_true, y_pred = y, knn_model_1.predict(X)
# print(classification_report(y_true, y_pred))
#
#
#
# c_table=confusion_matrix(y_true,y_pred)
#
#
# print('Confusion Matrix : \n', c_table )




# from sklearn.linear_model import LogisticRegression
#
#
# data = pd.read_csv("data.csv")
#
#
# y = data.iloc[:,0].values  # numpy array
# X = data.iloc[:,1:].values #features
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
#
# Logistic = LogisticRegression(C=1e5, solver='lbfgs').fit(X_train,y_train)
#
# Logistic.predict(X_train)
#
# print Logistic.score(X_train, y_train)
