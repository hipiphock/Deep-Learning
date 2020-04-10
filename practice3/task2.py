import numpy as np
import random
import math
import time

# KEY: just do W[2]a[1] for one hidden layer,
# INITAILIZATION
MIN_NUMBER = 1e-6

m = 1000
n = 100

alpha = 0.028    # learning rate
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
        x1 = random.randint(-2, 2)
        x2 = random.randint(-2, 2)
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
    global W1, W2, b1, b2, m, n
    batch_dW1 = np.array([0, 0])
    batch_dW2 = 0
    batch_db1 = 0
    batch_db2 = 0
    for Xi, Yi in zip(X, Y):
        ### forwarding ###
        # for layer 1 
        z1 = np.dot(W1, Xi) + b1
        a1 = sigmoid(z1)
        a1 = max(a1, MIN_NUMBER)        # to avoid divide by zero
        a1 = min(a1, 1- MIN_NUMBER)     # to avoid divide by zero
        # for layer 2
        z2 = np.dot(W2, a1) + b2        
        a2 = sigmoid(z2)
        a2 = max(a2, MIN_NUMBER)        # to avoid divide by zero
        a2 = min(a2, 1- MIN_NUMBER)     # to avoid divide by zero

        ### back propagation ###
        # need to find out dL/dW1, dL/dW2, dL/db1, dL/db2
        # 1. dL/db2
        dL_db2 = 1
        # 2. dL/dW2
        dL_da2 = -Yi/a2 + (1-Yi)/(1-a2)
        da2_dz2 = a2*(1-a2)
        dz2_dW2 = a1
        dL_dW2 = dL_da2*da2_dz2*dz2_dW2
        # 3. dL/dW1
        dz2_da1 = W2
        da1_dz1 = a1*(1-a1)
        dz1_dW1 = Xi
        dL_dW1 = dL_da2*da2_dz2*dz2_da1*da1_dz1*dz1_dW1
        # 4. dL/db1
        dz1_db1 = 1
        dL_db1 = dL_da2*da2_dz2*dz2_da1*da1_dz1*dz1_db1

        batch_dW1 = batch_dW1 + dL_dW1/m
        batch_dW2 = batch_dW2 + dL_dW2/m
        batch_db1 = batch_db1 + dL_db1/m
        batch_db2 = batch_db2 + dL_db2/m

    W1 = W1 - alpha*batch_dW1
    W2 = W2 - alpha*batch_dW2
    b1 = b1 - alpha*batch_db1
    b2 = b2 - alpha*batch_db2

def forward(Xi):
    global W1, W2, b1, b2, MIN_NUMBER
    z1 = np.dot(W1, Xi) + b1
    a1 = sigmoid(z1)
    a1 = max(a1, MIN_NUMBER)
    a1 = min(a1, 1 - MIN_NUMBER)
    z2 = np.dot(W2, z1) + b2
    a2 = sigmoid(z2)
    return a2

def accuracy(X, Y):
    num_correct = 0
    for i in range(len(X)):
        if Y[i] == round(forward(X[i])):
            num_correct += 1
    return num_correct/len(X)

def print_vectorized_w_b():
    print('w1: {}, w2: {}, b: {}'.format(W1[0], W1[1], b))

if __name__ == '__main__':
    train_X, train_Y = generate_data(m)
    test_X, test_Y = generate_data(n)
    start = time.time()
    for i in range(m):
        train(train_X, train_Y)
        # print("Iteration: "+ i.__str__())
    end = time.time()
    print('Time elapsed: ' + str(end - start) + 's')
    print('W1_1: {}, W1_2: {}, b1: {}'.format(W1[0], W1[1], b1))
    print('W2: {}, b2: {}'.format(W2, b2))
    print('train accuracy: {}'.format(accuracy(train_X, train_Y)))
    print('test accuracy: {}'.format(accuracy(test_X, test_Y)))