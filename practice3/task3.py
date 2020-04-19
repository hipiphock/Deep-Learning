# task3.py
import numpy as np
import random
import time

## INITIALIZATION
TRAIN_NUM = 1000
TEST_NUM = 100
alpha = 0.003   # learning rate

# initialize the variable
W1 = np.array([[0, 0], [0, 0], [0, 0]])     # 3*2 array
W2 = np.array([0, 0, 0])                    # 3*1 array
B1 = 0                                      # 3*1 array
B2 = 0

# generate random data
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
    global W1, W2, B1, B2, alpha, TRAIN_NUM, TEST_NUM
    # forwarding: layer 1
    Z1 = np.dot(W1, X) + B1
    A1 = sigmoid(Z1)
    # forwarding: layer 2
    Z2 = np.dot(np.transpose(W2), A1) + B2
    A2 = sigmoid(Z2)
    # back propagation: dW2
    dZ2 = A2 - Y                    # dL/dZ2
    dW2 = np.dot(A1, np.transpose(Z2))/TRAIN_NUM
    # back propagation: dB2
    dB2 = np.sum(dZ2)/TRAIN_NUM     
    # back propagation: dW1 - TODO
    # dZ2/dA1=W2, dA1/dZ1=A1(1-A1)
    print(np.shape(dZ2))
    print(np.shape(W2))
    print(np.shape(A1*(1-A1)))
    # dW1 = dZ2*W2*A1*(1-A1)   # (1000*1)*(1*3)*(3*1000)?
    tmp_arr = W2*np.dot(dZ2, np.transpose(A1*(1-A1)))
    # dW1 = np.dot(X, np.transpose(dZ2*W2*A1_1_A1))/TRAIN_NUM
    dW1 = np.dot(tmp_arr, np.transpose(X))/TRAIN_NUM
    # back propagation: dB1
    dB1 = dZ2*W2*A1*(1-A1)*(1-Z2)/TRAIN_NUM

    W1 = W1 - alpha*np.transpose(dW1)
    W2 = W2 - alpha*dW2
    B1 = B1 - alpha*dB1
    B2 = B2 - alpha*dB2

def forward(Xi):
    ### forwarding
    # for the first layer
    Z1 = np.dot(np.transpose(W1), X) + B1
    A1 = np.array(list(map(sigmoid, Z1)))
    # for the second layer
    Z2 = np.dot(W2, A1) + B2
    A2 = sigmoid(Z2)
    return A2

def accuracy(X, Y):
    num_correct = 0
    # forwarding: layer 1
    Z1 = np.dot(W1, X) + B1
    A1 = sigmoid(Z1)
    # forwarding: layer 2
    Z2 = np.dot(np.transpose(W2), A1) + B2
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
    print('W1: {}'.format(W1))
    print('B1: {}'.format(B1))
    print('W2: {}'.format(W2))
    print('B2: {}'.format(B2))
    print('train accuracy: {}'.format(accuracy(train_X, train_Y)))
    print('test accuracy: {}'.format(accuracy(test_X, test_Y)))