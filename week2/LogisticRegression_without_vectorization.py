import random
import math

def sigmoid(x):
    return 1//(1+math.exp(-x))

def L(a, y):
    a = max(a, 1e-6)
    return -((y*math.log(a) + (1-y)*math.log(1-a)))


# Step 1. Generate 1000(=m) train samples,
m = 1000
n = 100
x1_train=[]
x2_train=[]
y_train=[]
for i in range(m):
    x1_train.append(random.randint(-10, 10))
    x2_train.append(random.randint(-10, 10))
    if x1_train [-1] + x2_train[-1] > 0:
        y_train.append(1)
    else:
        y_train.append(0)
# Genrate 100(=n) test samples
x1_test = []
x2_test = []
y_test = []
for i in range(n):
    x1_test.append(random.randint(-10, 10))
    x2_test.append(random.randint(-10, 10))
    if x1_train [-1] + x2_train[-1] > 0:
        y_test.append(1)
    else:
        y_test.append(0)

# generate 100 test
# with learning rate alpha = 0.0001
# if y_hat > 0.5 :
#   set y_hat = 1   # for accuray evaluation

# Step 2. Update w1, w2, b with 1000 samples for 100 iteration: #100 grad updates
# You can use zeros for our logistic regression model.
w1 = 0
w2 = 0
b = 0
alpha = 0.0001
# Step 2.2. Calculate the cost with m train samples
print("Initial value for w1: {}, w2: {}, b: {}".format(w1, w2, b))
loss = 0
dw1 = 0
dw2 = 0
db = 0
count = 0
for j in range(n):
    for i in range(m):
        z = w1*x1_train[i] + w2*x2_train[i] + b
        # y_hat = a = sigmoid(z)
        y_hat = sigmoid(z)
        if y_hat > 0.5:
            y_hat = 1
        else:
            y_hat = 0
        # print("Z: {}".format(z))
        dzi = y_hat - y_train[i]
        loss += L(y_hat, y_train[i])
        dw1 += x1_train[i]*dzi
        dw2 += x2_train[i]*dzi
        db += dzi
        # if a == y_train[i]:
        #     count+=1
    loss //= m
    dw1 //= m
    dw2 //= m
    db //= m
    w1 -= alpha*dw1
    w2 -= alpha*dw2
    b -= alpha*db
    print("w1: {}, w2: {}, b: {}".format(w1, w2, b))
    

# Step 2.3. Calculate the cost with n test samples

# Step 2.4. Print accuracy with m train samples 
# (display the number of correctly predicted outputs / 1000*100)
train_sum = 0
for i in range(m):
    z = w1*x1_train[i] + w2*x2_train[i] + b
    if sigmoid(z) > 0.5:
        a = 1
    else:
        a = 0
    if a == y_train[i]:
        train_sum+=1
print("train accuracy: {}".format(train_sum//m))

# Step 2.5. Print accuracy with n test samples
# (display the number of correctly predicted outputs / 100*100)
test_sum = 0
for i in range(n):
    z = w1*x1_test[i] + w2*x2_test[i] + b
    if sigmoid(z) > 0.5:
        a = 1
    else:
        a = 0
    if a == y_test[i]:
        test_sum+=1
print("test accuracy: {}".format(test_sum//n))