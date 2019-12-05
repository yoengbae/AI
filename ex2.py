import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

name_file = './data_lab2_iis.txt'
columns = ['x', 'y']
data_in = pd.read_csv(name_file, names=columns, sep=' ')
data_in.plot(kind='scatter', x='x', y='y', color='red')

x = np.asarray(data_in['x'])
y = np.asarray(data_in['y'])

i = len(x)*7//10
training_x = x[:i]
training_y = y[:i]
test_x = x[i:]
test_y = y[i:]


def cfs(_x, _y):
    return np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(_x), _x)), np.transpose(_x)), _y)


lamb = 1.5
x_sqr = np.square(training_x)

_training_x = np.ones((len(training_x), 2))
_training_x[:, 1] = training_x
thetaA = cfs(_training_x, training_y)

__training_x = np.ones((len(training_x), 3))
__training_x[:, 1] = training_x
__training_x[:, 2] = np.square(training_x)
thetaB = cfs(__training_x, training_y)

___training_x = np.ones((len(training_x), 6))
tmp = training_x
for i in range(1, 6):
    ___training_x[:, i] = tmp
    tmp = np.multiply(tmp, training_x)
thetaC = cfs(___training_x, training_y)

n = np.identity(6)
n[0][0] = 0
x_t = np.transpose(___training_x)
thetaD = np.dot(np.dot(np.linalg.inv(np.dot(x_t, ___training_x)+lamb*n), x_t), training_y)

predict1 = np.dot(_training_x, thetaA)
predict2 = np.dot(__training_x, thetaB)
predict3 = np.dot(___training_x, thetaC)
predict4 = np.dot(___training_x, thetaD)

plt.figure(5)
plt.plot(training_x, training_y, 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(training_x, predict1, 'r')
plt.plot(training_x, predict2, 'g')
plt.plot(training_x, predict3, 'b')
plt.plot(training_x, predict4, 'c')
print("training")
print("unregularized linear", sum(abs(training_y-predict1)))
print("unregularized parabolic", sum(abs(training_y-predict2)))
print("unregularized 5th", sum(abs(training_y-predict3)))
print("regularized 5th", sum(abs(training_y-predict4)))

_test_x = np.ones((len(test_x), 2))
_test_x[:, 1] = test_x
__test_x = np.ones((len(test_x), 3))
__test_x[:, 1] = test_x
__test_x[:, 2] = np.square(test_x)
___test_x = np.ones((len(test_x), 6))
tmp = test_x
for i in range(1, 6):
    ___test_x[:, i] = tmp
    tmp = np.multiply(tmp, test_x)

predict1_test = np.dot(_test_x, thetaA)
predict2_test = np.dot(__test_x, thetaB)
predict3_test = np.dot(___test_x, thetaC)
predict4_test = np.dot(___test_x, thetaD)
plt.plot(test_x, test_y, 'go')
plt.plot(test_x, predict1_test, 'r')
plt.plot(test_x, predict2_test, 'g')
plt.plot(test_x, predict3_test, 'b')
plt.plot(test_x, predict4_test, 'c')

print("test")
print("unregularized linear", sum(abs(test_y-predict1_test)))
print("unregularized parabolic", sum(abs(test_y-predict2_test)))
print("unregularized 5th", sum(abs(test_y-predict3_test)))
print("regularized 5th", sum(abs(test_y-predict4_test)))

plt.show()
