import numpy as np
import pandas as pd
from sklearn import preprocessing
from NeuralNetwork import NeuralNetwork
from statistic import *
#from sklearn.metrics import accuracy_score, f1_score

train = pd.read_csv('./dm_nccu_2020_fall/training.csv')
test = pd.read_csv('./dm_nccu_2020_fall/testing.csv')

encoder = preprocessing.LabelEncoder()

x_train = train.drop(columns=['Attrition', 'EmployeeNumber'])
for key in x_train.keys():
    _label = encoder.fit_transform(train[key])
    x_train[key] = _label
x_train = x_train.to_numpy()

y_train = train[['Attrition']]
for key in y_train.keys():
    _label = encoder.fit_transform(train[key])
    y_train[key] = _label
y_train = y_train.to_numpy()

x_test = test.drop(columns=['Attrition', 'EmployeeNumber'])
for key in x_test.keys():
    _label = encoder.fit_transform(test[key])
    x_test[key] = _label
x_test = x_test.to_numpy()

y_test = test[['Attrition']]
for key in y_test.keys():
    _label = encoder.fit_transform(test[key])
    y_test[key] = _label
y_test = y_test.to_numpy()


NN = NeuralNetwork()
NN.train(x_train, y_train, 20000)

y_pred = NN.predict(x_test)
y_pred = toBinary(y_pred)

print('=' * 50)
print('Accuracy:', accuracy(y_pred, y_test))
print('Precision:', precision(y_pred, y_test))
print('Recall:', recall(y_pred, y_test))
print('F1 Score:', f1_score(y_pred, y_test))
#print('Accuracy:', accuracy_score(y_test, y_pred))
#print('F1 Score:', f1_score(y_test, y_pred, average='macro'))


