import numpy as np
import random
import time

# KEY: just do W[2]a[1] for one hidden layer,
# INITAILIZATION
MIN_NUMBER = 1e-6

TRAIN_NUM = 1000
TEST_NUM = 100

alpha = 0.00005    # learning rate
loss = 0        

W1 = np.array([0, 0])
W2 = 0
b1 = 0
b2 = 0

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
def train(X, Y):
    global W1, W2, b1, b2, TRAIN_NUM, TEST_NUM
    # layer 1, Z1 is vector, A1 is number
    Z1 = np.dot(np.transpose(W1), X) + b1
    A1 = sigmoid(Z1)
    # layer 2, Z2 is number, A2 is number
    Z2 = W2*A1 + b2
    A2 = sigmoid(Z2)
    # dL/dW2
    dZ2 = A2 - Y
    dW2 = np.dot(A1, np.transpose(dZ2))/TRAIN_NUM
    # dL/db2
    dB2 = np.sum(dZ2)/TRAIN_NUM
    # dL/dW1
    dA1_dZ1 = A1*(1-A1)
    dZ1_dW1 = X
    dW1 = np.dot(X, dZ2*W2*dA1_dZ1)/TRAIN_NUM
    # dL/db1
    dB1 = np.sum(dZ2*W2*dA1_dZ1)/TRAIN_NUM

    W1 = W1 - alpha*dW1
    W2 = W2 - alpha*dW2
    b1 = b1 - alpha*dB1
    b2 = b2 - alpha*dB2


def loss(X, Y):
    global W1, W2, b1, b2, TRAIN_NUM, TEST_NUM
    batch_loss = 0
    Z1 = np.dot(np.transpose(W1), X) + b1
    A1 = sigmoid(Z1)
    Z2 = W2*A1 + b2
    A2 = sigmoid(Z2)
    B_L = -(Y*np.log(A2) + (1-Y)*np.log(1-A2))
    batch_loss = np.sum(B_L)/TRAIN_NUM
    return batch_loss

def accuracy(X, Y):
    num_correct = 0
    Z1 = np.dot(np.transpose(W1), X) + b1
    A1 = sigmoid(Z1)
    Z2 = W2*A1 + b2
    A2 = sigmoid(Z2)
    for Ai, Yi in zip(A2, Y):
        if round(Ai) == Yi:
            num_correct += 1
    return num_correct/len(np.transpose(X))

if __name__ == '__main__':
    train_X, train_Y = generate_data(TRAIN_NUM)
    test_X, test_Y = generate_data(TEST_NUM)
    start = time.time()
    for i in range(TRAIN_NUM):
        train(train_X, train_Y)
    end = time.time()
    print('Time elapsed: ' + str(end - start) + 's')
    print('W1: {}, b1: {}'.format(W1, b1))
    print('W2: {}, b2: {}'.format(W2, b2))
    print('train loss: {}, accuracy: {}'.format(loss(train_X, train_Y), accuracy(train_X, train_Y)))
    print('test loss: {}, accuracy: {}'.format(loss(test_X, test_Y), accuracy(test_X, test_Y)))