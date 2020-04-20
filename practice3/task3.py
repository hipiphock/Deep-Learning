# task3.py
import numpy as np
import random
import time

## INITIALIZATION
TRAIN_NUM = 1000
TEST_NUM = 100
alpha = 0.003   # learning rate

# initialize the variable
W1 = np.zeros((3,2))
b1 = np.zeros((3,1))
W2 = np.zeros((1,3))
b2 = np.zeros((1,1))

# generate random data
# X, Y are ()*1000
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

def train(X, Y):
    global W1, W2, b1, b2, alpha, TRAIN_NUM, TEST_NUM
    # forwarding: layer 1
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    # forwarding: layer 2
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    # back propagation: dW2
    dZ2 = A2 - Y                # dL/dZ2
    dW2 = np.dot(Z2, np.transpose(A1))/TRAIN_NUM
    # back propagation: dB2
    dB2 = np.sum(dZ2)/TRAIN_NUM
    # back propagation: dW1
    dZ1 = np.dot(np.transpose(W2), dZ2)*A1*(1-A1)
    dW1 = np.dot(dZ1, np.transpose(X))/TRAIN_NUM
    # back propagation: dB1
    dB1 = np.sum(dZ1)/TRAIN_NUM

    W1 = W1 - alpha*dW1
    W2 = W2 - alpha*dW2
    b1 = b1 - alpha*dB1
    b2 = b2 - alpha*dB2


def forward(X):
    ### forwarding
    # for the first layer
    Z1 = np.dot(np.transpose(W1), X) + b1
    A1 = np.array(list(map(sigmoid, Z1)))
    # for the second layer
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    return A2

def accuracy(X, Y):
    num_correct = 0
    # forwarding: layer 1
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    # forwarding: layer 2
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    A2[A2 > 0.5] = 1
    A2[A2 <= 0.5] = 0
    num_correct = np.sum(A2 == Y)
    return num_correct/len(np.transpose(X))

if __name__ == '__main__':
    train_X, train_Y = generate_data(TRAIN_NUM)
    test_X, test_Y = generate_data(TEST_NUM)
    start = time.time()
    for i in range(TRAIN_NUM):
        train(train_X, train_Y)
    end = time.time()
    print('Time elapsed: ' + str(end - start) + 's')
    print('W1: {}'.format(W1))
    print('B1: {}'.format(b1))
    print('W2: {}'.format(W2))
    print('B2: {}'.format(b2))
    print('train accuracy: {}'.format(accuracy(train_X, train_Y)))
    print('test accuracy: {}'.format(accuracy(test_X, test_Y)))