import numpy as np

def toBinary(x):
    for i in range(x.shape[0]):
        if x[i][0] < 0.5:
            x[i][0] = 0
        else:
            x[i][0] = 1
    return x

def accuracy(x, y):
    count = 0
    for i in range(x.shape[0]):
        if x[i][0] == y[i][0]:
            count += 1
    return count / x.shape[0]

def precision(prediction, answer):
    TP = 0
    FP = 0
    for i in range(prediction.shape[0]):
        if prediction[i][0] == 1 and answer[i][0] == 1:
            TP += 1
        if prediction[i][0] == 1 and answer[i][0] == 0:
            FP += 1
    return TP / (TP + FP)

def recall(prediction, answer):
    TP = 0
    FN = 0
    for i in range(prediction.shape[0]):
        if prediction[i][0] == 1 and answer[i][0] == 1:
            TP += 1
        if prediction[i][0] == 0 and answer[i][0] == 1:
            FN += 1
    return TP / (TP + FN)

def f1_score(prediction, answer):
    return 2 * precision(prediction, answer) * recall(prediction, answer) / (precision(prediction, answer) + recall(prediction, answer))
