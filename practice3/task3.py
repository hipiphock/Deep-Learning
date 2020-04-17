# task3.py
import numpy as np
import random
import math
import time

## INITIALIZATION
# 정의해야 할 것:
# input x의 dimension
# layer의 수,
# 각 layer의 node의 수
TRAIN_NUM = 1000
TEST_NUM = 100
# initialize the variable
W1 = np.array([[0, 0], [0, 0], [0, 0]])
W2 = np.array([0, 0, 0])
B1 = np.array([0, 0, 0])
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
    return X, Y

# Sigmoid function
def sigmoid(val):
    return 1/(1+math.exp(-val))

def train(X, Y):
    global W1, W2, B1, B2
    dW1 = np.array([[0, 0], [0, 0], [0, 0]])
    dW2 = np.array([0, 0, 0])
    dB1 = np.array([0, 0, 0])
    dB2 = 0
    # train each datum from train_data_X and train_data_Y
    for Xi, Yi in zip(X, Y):
        ### forwarding
        # for the first layer
        Z1 = np.dot(W1, Xi) + B1
        A1 = list(map(sigmoid, Z1))
        # for the second layer
        Z2 = np.dot(W2, A1) + B2
        A2 = sigmoid(Z2)
        ### back propagation
        # need to find out dL/dW2, dL/dW1, dL/dB2. dL/dB1
        dL_dW2 = list(map(lambda x: x*(A2-Yi), A1))
        dL_dB2 = (A2-Yi)*1
        dL_dW1 = (A2-Yi)*(Z2*(1-Z2))*(A1*(1-A1))*(Xi)
        dL_dB1 = (A2 - Yi) * (Z2*(1 - Z2)) * (A1*(1 - A1)) * (B1)
        dW1 = dW1 + dL_dW1
        dW2 = dW2 + dL_dW2
        dB1 = dB1 + dL_dB1
        dB2 = dB2 + dL_dB2
    dW1 = dW1 / TRAIN_NUM
    dW2 = dW2 / TRAIN_NUM
    dB1 = dB1 / TRAIN_NUM
    dB2 = dB2 / TRAIN_NUM
    return dW1, dW2, dB1, dB2
           
if __name__ == '__main__':
    train_X, train_Y = generate_data(TRAIN_NUM)
    test_X, test_Y = generate_data(TEST_NUM)
    start = time.time()
    for i in range(TRAIN_NUM):
        train(train_X, train_Y)
    end = time.time()