# task3.py
import numpy as np
import random
import time

## INITIALIZATION
TRAIN_NUM = 1000
TEST_NUM = 100
alpha = 1   # learning rate

# initialize the variable
W1 = np.random.randn(3,2)
b1 = np.random.randn(3,1)
W2 = np.random.randn(1,3)
b2 = np.random.randn(1,1)
# W1 = np.zeros((3,2))
# b1 = np.zeros((3,1))
# W2 = np.zeros((1,3))
# b2 = np.zeros((1,1))

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
    # forwarding: layer 1
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    # forwarding: layer 2
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    return A2

def loss(X, Y):
    loss = 0
    A2 = forward(X)
    L = -(Y*np.log(A2) + (1-Y)*np.log(1-A2))
    loss = np.sum(L) / len(Y)
    return loss

def accuracy(X, Y):
    num_correct = 0
    A2 = forward(X)
    A2[A2 > 0.5] = 1
    A2[A2 <= 0.5] = 0
    num_correct = np.sum(A2 == Y)
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
    start = time.time()
    # find_best_alpha(train_X, train_Y)
    for i in range(TRAIN_NUM):
        train(train_X, train_Y)
    end = time.time()
    print('Time elapsed: ' + str(end - start) + 's')
    print('W1: {}'.format(W1))
    print('B1: {}'.format(b1))
    print('W2: {}'.format(W2))
    print('B2: {}'.format(b2))
    print('train loss: {}, accuracy: {}'.format(loss(train_X, train_Y), accuracy(train_X, train_Y)))
    print('test loss: {}, accuracy: {}'.format(loss(test_X, test_Y), accuracy(test_X, test_Y)))