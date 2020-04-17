# task3.py
import numpy as np
import random
import time

## INITIALIZATION
TRAIN_NUM = 1000
TEST_NUM = 100
alpha = 0.003   # learning rate

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
    return 1/(1+np.exp(-val))

def train(X, Y):
    global W1, W2, B1, B2, alpha
    batch_dW1 = np.array([[0, 0], [0, 0], [0, 0]])
    batch_dW2 = np.array([0, 0, 0])
    batch_dB1 = np.array([0, 0, 0])
    batch_dB2 = 0
    # train each datum from train_data_X and train_data_Y
    for Xi, Yi in zip(X, Y):
        ### forwarding
        # for the first layer
        Z1 = np.dot(W1, Xi) + B1
        A1 = np.array(list(map(sigmoid, Z1)))
        # for the second layer
        Z2 = np.dot(W2, A1) + B2
        A2 = sigmoid(Z2)
        ### back propagation
        # need to find out dL/dW2, dL/dW1, dL/dB2. dL/dB1
        dL_dW2 = np.array(list(map(lambda x: x*(A2-Yi), A1)))
        dL_dB2 = (A2-Yi)*1
        # TODO: need to handle A1(1-A1)
        A1_1_A1 = np.array(list(map(lambda x: x*(1-x), A1)))
        tmp_array = np.array(list(map(lambda x: x*(A2-Yi)*(Z2*(1-Z2)), A1_1_A1)))
        Xi_multiplied = np.array([Xi, Xi, Xi])
        dL_dW1 = np.dot(tmp_array, Xi_multiplied)
        dL_dB1 = np.array(list(map(lambda x: x*(A2 - Yi) * (Z2*(1 - Z2)), A1_1_A1)))

        batch_dW1 = batch_dW1 + dL_dW1/TRAIN_NUM
        batch_dW2 = batch_dW2 + dL_dW2/TRAIN_NUM
        batch_dB1 = batch_dB1 + dL_dB1/TRAIN_NUM
        batch_dB2 = batch_dB2 + dL_dB2/TRAIN_NUM
        
    W1 = W1 - alpha*batch_dW1
    W2 = W2 - alpha*batch_dW2
    B1 = B1 - alpha*batch_dB1
    B2 = B2 - alpha*batch_dB2

def forward(Xi):
    ### forwarding
    # for the first layer
    Z1 = np.dot(W1, Xi) + B1
    A1 = np.array(list(map(sigmoid, Z1)))
    # for the second layer
    Z2 = np.dot(W2, A1) + B2
    A2 = sigmoid(Z2)
    return A2

def accuracy(X, Y):
    num_correct = 0
    for i in range(len(X)):
        if Y[i] == round(forward(X[i])):
            num_correct += 1
    return num_correct/len(X)

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