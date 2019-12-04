import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

name_file = './data_lab1_iis.txt'

columns = ['x', 'y']

data_in = pd.read_csv(name_file, names=columns, sep=' ')

data_in.plot(kind='scatter', x='x', y='y', color='red')

x = np.asarray(data_in['x'])
y = np.asarray(data_in['y'])

plt.figure(5)
plt.plot(x, y, 'ro')
plt.xlabel('x')
plt.ylabel('y')


################1)


def y_hat(__x, _theta):
    _y = np.dot(np.transpose(_theta), __x)
    return _y


_x = np.ones((len(x), 2))
_x[:, 1] = x
alpha = 0.01
theta1 = np.array([[2], [2]], dtype=float)

for i in range(100):
    for q in range(len(x)):
        theta1[0] = theta1[0] - alpha * (y_hat(_x[q], theta1) - y[q])
        theta1[1] = theta1[1] - alpha * (y_hat(_x[q], theta1) - y[q]) * x[q]

################2)BGD

theta2 = np.array([[2], [2]], dtype=float)

for i in range(100):
    r = random.randint(0, len(x)-1)
    theta2[0] = theta2[0] - alpha * (y_hat(_x[r], theta2) - y[r])
    theta2[1] = theta2[1] - alpha * (y_hat(_x[r], theta2) - y[r]) * x[r]

################3)SGD

x_t = np.transpose(_x)
tm = np.linalg.inv(np.dot(x_t, _x))
theta3 = np.dot(np.dot(tm, x_t), y)

################4)CFS

y1 = [theta1[0]+theta1[1]*v for v in x]
y2 = [theta2[0]+theta2[1]*v for v in x]
y3 = [theta3[0]+theta3[1]*v for v in x]
plt.plot(x, y1, 'bo')   # BGD, blue line
plt.plot(x, y2, 'go')   # SGD, green line
plt.plot(x, y3, 'co')   # CFS, cyan line

################5)

plt.plot(3, theta1[0]+theta1[1]*3, 'bo')
plt.plot(3, theta2[0]+theta2[1]*3, 'go')
plt.plot(3, theta3[0]+theta3[1]*3, 'co')

plt.show()

################6)
