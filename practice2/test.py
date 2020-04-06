# import LogisticRegressionWithoutVectorization
# import LogisticRegressionWithVectorization

from LogisticRegressionWithVectorization import * 
from LogisticRegressionWithoutVectorization import * 

import numpy as np
import random
import math
import time
from typing import List, Any

TRAIN_NUM = 1000
TEST_NUM = 100

# data generator
def generate_data(size):
    X = []
    Y = []
    datalist = []
    for i in range(size):
        x1 = random.randint(-10, 10)
        x2 = random.randint(-10, 10)
        data = {'x1':x1, 'x2':x2}
        if x1 + x2 > 0:
            Y.append(1)
            data['y'] = 1
        else:
            Y.append(0)
            data['y'] = 0
        X.append(np.array([x1, x2]))
        datalist.append(data)
    return X, Y, datalist

if __name__ == '__main__':
    train_X, train_Y, train_data = generate_data(TRAIN_NUM)
    test_X, test_Y, test_data = generate_data(TEST_NUM)
    
    print("Logistic Regression Without Vectorizaiton:")
    start = time.time()
    for i in range(TRAIN_NUM):
        train_unvectorized(train_data)
    end = time.time()
    print('train_unvectorized loss: {}, accuracy: {}'.format(loss_without_vectorization(train_data), accuracy_without_vectorization(train_data)))
    print('test loss: {}, accuracy: {}'.format(loss_without_vectorization(test_data), accuracy_without_vectorization(test_data)))
    print_unvectorized_w_b()
    print('Time elapsed: ' + str(end - start) + 's')

    print("Logistic Regression With Vectorizaiton:")
    start = time.time()
    for i in range(TRAIN_NUM):
        train_vectorized(train_X, train_Y)
    end = time.time()
    print('train_vectorized loss: {}, accuracy: {}'.format(loss_with_vectorization(train_X, train_Y), accuracy_with_vectorization(train_X, train_Y)))
    print('test loss: {}, accuracy: {}'.format(loss_with_vectorization(test_X, test_Y), accuracy_with_vectorization(test_X, test_Y)))
    print_vectorized_w_b()
    print('Time elapsed: ' + str(end - start) + 's')