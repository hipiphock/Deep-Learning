import random
import math
from typing import List, Any

# INITAILIZATION
MIN_NUMBER = 1e-6

m = 1000
n = 100

alpha = 0.0001 # learning rate
loss = 0

w1, w2, b = 0, 0, 0

# data generator
def generate_random_data(size):
    datalist = []
    for i in range(size):
        data = {'x1':random.randint(-10, 10), 'x2':random.randint(-10, 10)}
        if data['x1'] + data['x2'] > 0:
            data['y'] = 1
        else:
            data['y'] = 0
        datalist.append(data)
    return datalist

# Sigmoid function
def sigmoid(x):
    return 1/(1+math.exp(-x))

# Loss function for Logistic Regression
def L(a, y):
    return -(y*math.log(a) + (1-y)*math.log(1-a))

# training function
def train_unvectorized(datalist):
    global w1, w2, b
    batch_dw1, batch_dw2, batch_db = 0, 0, 0
    for data in datalist:
        z = w1*data['x1'] + w2*data['x2'] + b
        a = sigmoid(z)
        da = -data['y']/a + (1-data['y'])/(1-a)
        dz = da * a * (1-a)
        dw1 = data['x1']*dz
        dw2 = data['x2']*dz
        db = dz
        batch_dw1 += dw1 / len(datalist)
        batch_dw2 += dw2 / len(datalist)
        batch_db += db / len(datalist)
    w1 -= alpha*batch_dw1
    w2 -= alpha*batch_dw2
    b -= alpha*batch_db

def forward(x1, x2):
    global w1, w2, b
    z = w1 * x1 + w2 * x2 + b
    a = sigmoid(z)
    MIN_VAL = 1e-10
    a = max(a, MIN_VAL)
    a = min(a, 1 - MIN_VAL)
    return a

def loss_without_vectorization(datalist):
    batch_loss = 0
    for data in datalist:
        pred_y = forward(data['x1'], data['x2'])
        batch_loss -= data['y'] * math.log(pred_y) + (1 - data['y']) * math.log(1 - pred_y)
    batch_loss /= len(datalist)
    return batch_loss

def accuracy_without_vectorization(datalist):
    num_correct = 0
    for data in datalist:
        z = w1*data['x1'] + w2*data['x2'] + b
        a = sigmoid(z)
        if data['y'] == round(forward(data['x1'], data['x2'])):
            num_correct += 1
    return num_correct/len(datalist)

def print_unvectorized_w_b():
    print('w1: {}, w2: {}, b: {}'.format(w1, w2, b))

# if __name__ == '__main__':
#     train_data = generate_random_data(m)
#     test_data = generate_random_data(n)
#     for i in range(m):
#         train_unvectorized(train_data)
#         print("Iteration: "+ i.__str__())
#         print('w1: {}, w2: {}, b: {}'.format(w1, w2, b))
#         print('train_unvectorized loss: {}, accuracy: {}'.format(loss(train_data), accuracy(train_data)))
#         print('test loss: {}, accuracy: {}'.format(loss(test_data), accuracy(test_data)))