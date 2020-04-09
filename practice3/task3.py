# task3.py
import numpy as np
import random
import math
from typing import List, Any

# INITIALIZATION
W1 = np.array([[0, 0], [0, 0], [0, 0]])
# Z1 = np.array([0, 0, 0])
W2 = np.array([0, 0, 0])
# Z2 = 0
B1 = np.array([0, 0, 0])
B2 = 0

# data generator
def generate_random_data(size):
    X = []
    Y = []
    for i in range(size):
        x1 = random.randint(-2, 2)
        x2 = random.randint(-2, 2)
        if x1 + x2 > 0:
            Y.append(1)
        else:
            Y.append(0)
        X.append(np.array([x1, x2]))
    return X, Y

def train():
    global W1, W2, B1, B2

if __name__ == '__main__':
    train_X, train_Y = generate_random_data(m)
    test_X, test_Y = generate_random_data(n)
    for i in range(m):
        train(train_X, train_Y)
        # print("Iteration: "+ i.__str__())
    print('W1_1: {}, W1_2: {}, W2: {}, b1: {}, b2: {}'.format(W1[0], W1[1], W2, b1, b2))
    print('train accuracy: {}'.format(accuracy_with_vectorization(train_X, train_Y)))
    print('test accuracy: {}'.format(accuracy_with_vectorization(test_X, test_Y)))