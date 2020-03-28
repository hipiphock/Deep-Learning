import LogisticRegressionWithoutVectorization
import LogisticRegressionWithVectorization

import numpy as np
import random
import math
from typing import List, Any

# data generator
def generate_data_unvectorized(size):
    datalist = []
    for i in range(size):
        data = {'x1':random.randint(-10, 10), 'x2':random.randint(-10, 10)}
        if data['x1'] + data['x2'] > 0:
            data['y'] = 1
        else:
            data['y'] = 0
        datalist.append(data)
    return datalist

# data generator
def generate_data_vectorized(size):
    X = []
    Y = []
    for i in range(size):
        x1 = random.randint(-10, 10)
        x2 = random.randint(-10, 10)
        if x1 + x2 > 0:
            Y.append(1)
        else:
            Y.append(0)
        X.append(np.array([x1, x2]))
    return X, Y