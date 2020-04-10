# task3.py
import numpy as np
import random
import math
from typing import List, Any

## INITIALIZATION
# 정의해야 할 것:
# input x의 dimension
# layer의 수,
# 각 layer의 node의 수
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

