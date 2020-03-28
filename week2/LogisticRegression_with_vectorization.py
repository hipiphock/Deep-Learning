import numpy as np
import random
import math
from typing import List, Any

# INITAILIZATION
MIN_NUMBER = 1e-6

m = 1000
n = 100

alpha = 0.0001 # learning rate
loss = 0

w = [0, 0, 0]
b = 0

# data generator
def generate_random_data(size):
    datalist = []
    for i in range(size):
        data = []
        x1 = random.randint(-10, 10)
        x2 = random.randint(-10, 10)
        data.append(x1)
        data.append(x2)
        if data[0] + data[1] > 0:
            data.append(1)
        else:
            data.append(0)
        datalist.append(data)
    return datalist

# Sigmoid function
def sigmoid(x):
    return 1/(1+math.exp(-x))

# Loss function for Logistic Regression
def L(a, y):
    return -(y*math.log(a) + (1-y)*math.log(1-a))

# training function
def train(datalist):
    global w, b
    batch_dw1, batch_dw2, batch_db = 0, 0, 0
    for x in datalist:
        z = np.dot(w, x) + b
        a = sigmoid(z)
        da = -x[2]/a + (1-x[2])/(1-a)
        dz = da * a * (1-a)
        dw1 = x[0]*dz
        dw2 = x[1]*dz
        db = dz
        batch_dw1 += dw1 / len(datalist)
        batch_dw2 += dw2 / len(datalist)
        batch_db += db / len(datalist)
    w[0] -= alpha*batch_dw1
    w[1] -= alpha*batch_dw2
    b -= alpha*batch_db


def forward(x):
    global w, b
    z = np.dot(w, x) + b
    a = sigmoid(z)
    MIN_VAL = 1e-10
    a = max(a, MIN_VAL)
    a = min(a, 1 - MIN_VAL)
    return a

def loss(datalist):
    batch_loss = 0
    for x in datalist:
        pred_y = forward(x)
        batch_loss -= x[2] * math.log(pred_y) + (1 - x[2]) * math.log(1 - pred_y)
    batch_loss /= len(datalist)
    return batch_loss

def accuracy(datalist):
    num_correct = 0
    for x in datalist:
        z = z = np.dot(w, x) + b
        a = sigmoid(z)
        if x[2] == round(forward(x)):
            num_correct += 1
    return num_correct/len(datalist)

if __name__ == '__main__':
    train_data = generate_random_data(m)
    test_data = generate_random_data(n)
    for i in range(m):
        train(train_data)
        print("Iteration: "+ i.__str__())
        print('w1: {}, w2: {}, b: {}'.format(w[0], w[1], b))
        print('train loss: {}, accuracy: {}'.format(loss(train_data), accuracy(train_data)))
        print('test loss: {}, accuracy: {}'.format(loss(test_data), accuracy(test_data)))