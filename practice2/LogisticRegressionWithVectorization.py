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

W = np.array([0, 0])
b = 0

# data generator
def generate_random_data(size):
    X = []
    Y = []
    for i in range(size):
        x1 = random.randint(-10, 10)
        x2 = random.randint(-10, 10)
        if x1 + x2 > 0:
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
def train_vectorized(X, Y):
    global W, b
    batch_dW = np.array([0, 0])
    batch_db = 0
    # iteration with index
    # for i in range(len(X)):
    #     z = np.dot(W, X[i]) + b
    #     a = sigmoid(z)
    #     da = -Y[i]/a + (1-Y[i])/(1-a)
    #     dz = da * a * (1-a)
    #     dW = X[i]*dz
    #     db = dz
    #     batch_dW = batch_dW + dW/len(X)
    #     batch_db += db/len(X)
    # W = W - alpha*batch_dW
    # b -= alpha*batch_db
    # iteration using zip - have bug with it
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

def loss_with_vectorization(X, Y):
    batch_loss = 0
    for i in range(len(X)):
        pred_y = forward(X[i])
        batch_loss -= Y[i] * math.log(pred_y) + (1 - Y[i]) * math.log(1 - pred_y)
    batch_loss /= len(X)
    return batch_loss

def accuracy_with_vectorization(X, Y):
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
    for i in range(m):
        train_vectorized(train_X, train_Y)
        print("Iteration: "+ i.__str__())
        print('w1: {}, w2: {}, b: {}'.format(W[0], W[1], b))
        print('train_vectorized loss: {}, accuracy: {}'.format(loss_with_vectorization(train_X, train_Y), accuracy_with_vectorization(train_X, train_Y)))
        print('test loss: {}, accuracy: {}'.format(loss_with_vectorization(test_X, test_Y), accuracy_with_vectorization(test_X, test_Y)))