import numpy as np
import random
import math
from typing import List, Any
import time

# INITAILIZATION
MIN_NUMBER = 1e-6

m = 1000
n = 100

alpha = 0.003 # learning rate
loss = 0

W = np.array([0, 0])
b = 0

# data generator
def generate_random_data(size):
    X = []
    Y = []
    for i in range(size):
        x1 = random.uniform(-2, 2)
        x2 = random.uniform(-2, 2)
        if x1 * x1 > x2:
            Y.append(1)
        else:
            Y.append(0)
        X.append(np.array([x1, x2]))
    return X, Y

# Sigmoid function
def sigmoid(val):
    return 1/(1+math.exp(-val))

# Loss function for Logistic Regression
def L(a, y):
    return -(y*math.log(a) + (1-y)*math.log(1-a))

# training function
def train(X, Y):
    global W, b
    batch_dW = np.array([0, 0])
    batch_db = 0
    for Xi, Yi in zip(X, Y):
        z = np.dot(W, Xi) + b
        a = sigmoid(z)
        da = -Yi/a + (1-Yi)/(1-a)
        dz = da * a * (1-a)
        dW = Xi*dz
        db = dz
        batch_dW = batch_dW + dW/len(X)
        batch_db += db/len(X)
    W = W - alpha*batch_dW
    b -= alpha*batch_db

def forward(Xi):
    global W, b
    z = np.dot(W, Xi) + b
    a = sigmoid(z)
    MIN_VAL = 1e-10
    a = max(a, MIN_VAL)
    a = min(a, 1 - MIN_VAL)
    return a

def loss(X, Y):
    batch_loss = 0
    for i in range(len(X)):
        pred_y = forward(X[i])
        batch_loss -= Y[i] * math.log(pred_y) + (1 - Y[i]) * math.log(1 - pred_y)
    batch_loss /= len(X)
    return batch_loss

def accuracy(X, Y):
    num_correct = 0
    for i in range(len(X)):
        z = np.dot(W, X[i]) + b
        a = sigmoid(z)
        if Y[i] == round(forward(X[i])):
            num_correct += 1
    return num_correct/len(X)

def print_vectorized_w_b():
    print('w1: {}, w2: {}, b: {}'.format(W[0], W[1], b))

if __name__ == '__main__':
    train_X, train_Y = generate_random_data(m)
    test_X, test_Y = generate_random_data(n)
    start = time.time()
    for i in range(m):
        train(train_X, train_Y)
        # print("Iteration: "+ i.__str__())
    end = time.time()
    print('Time elapsed: ' + str(end - start) + 's')
    print('w1: {}, w2: {}, b: {}'.format(W[0], W[1], b))
    print('train loss: {}, accuracy: {}'.format(loss(train_X, train_Y), accuracy(train_X, train_Y)))
    print('test loss: {}, accuracy: {}'.format(loss(test_X, test_Y), accuracy(test_X, test_Y)))