import numpy as np
import random
from typing import List, Any
import time

# INITAILIZATION
MIN_NUMBER = 1e-6

TRAIN_NUM = 1000
TEST_NUM = 100

alpha = 0.005 # learning rate
loss = 0

W = np.array([0, 0])
b = 0

# data generator
def generate_data(size):
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
    return np.transpose(X), np.transpose(Y)

# Sigmoid function
def sigmoid(val):
    return 1/(1+np.exp(-val))

# Loss function for Logistic Regression
def L(a, y):
    return -(y*np.log(a) + (1-y)*np.log(1-a))

# training function
# X is 2*1000 array
# Y is 1*1000 array
def train(X, Y):
    global W, b
    # forwarding
    Z = np.dot(np.transpose(W), X) + b
    A = sigmoid(Z)
    # back propagation
    dZ = A - Y
    dW = np.dot(X, np.transpose(dZ))/TRAIN_NUM
    db = np.sum(dZ)/TRAIN_NUM
    
    W = W - alpha*dW
    b -= alpha*db

def loss(X, Y):
    batch_loss = 0
    Z = np.dot(np.transpose(W), X) + b
    A = sigmoid(Z)
    B_L = -(Y*np.log(A) + (1-Y)*np.log(1-A))
    batch_loss = np.sum(B_L)/TRAIN_NUM
    return batch_loss

def accuracy(X, Y):
    num_correct = 0
    Z = np.dot(np.transpose(W), X) + b
    A = sigmoid(Z)
    for Ai, Yi in zip(A, Y):
        if round(Ai) == Yi:
            num_correct += 1
    return num_correct/len(np.transpose(X))

def find_best_alpha(X, Y):
    global alpha
    # Using Ternary Search
    head = 0
    tail = 1.0
    cnt = 0
    best_alpha = 0.001
    while tail - head > 1e-8:
        p = (2 * head + tail) / 3
        q = (head + 2 * tail) / 3
        alpha = p
        train(X, Y)
        p_loss = loss(X, Y)
        alpha = q
        train(X, Y)
        q_loss = loss(X, Y)
        cnt += 1
        # print('%d Search: [%.6f, %.6f, %.6f, %.6f] => loss_p: %.6f, loss_q: %.6f' % (cnt, head, p, q, tail, p_loss, q_loss))/
        if p_loss > q_loss:
            head = p
            best_alpha = q
        elif p_loss <= q_loss:
            tail = q
            best_alpha = p
    print('Best Learning Rate: %.6f' % best_alpha)
    alpha = best_alpha

if __name__ == '__main__':
    train_X, train_Y = generate_data(TRAIN_NUM)
    test_X, test_Y = generate_data(TEST_NUM)
    find_best_alpha(train_X, train_Y)
    start = time.time()
    for i in range(TRAIN_NUM):
        train(train_X, train_Y)
    end = time.time()
    print('Time elapsed: ' + str(end - start) + 's')
    print('w1: {}, w2: {}, b: {}'.format(W[0], W[1], b))
    print('train loss: {}, accuracy: {}'.format(loss(train_X, train_Y), accuracy(train_X, train_Y)))
    print('test loss: {}, accuracy: {}'.format(loss(test_X, test_Y), accuracy(test_X, test_Y)))