import numpy as np
import pandas as pd

name_file = 'data_FFNN.txt'
columns = ['x1', 'x2', 'y']
data_in = pd.read_csv(name_file, names=columns, sep=' ')
data_in.plot(kind='scatter', x='x1', y='x2', color='red')
x1 = np.asarray(data_in['x1'])
x2 = np.asarray(data_in['x2'])
y = np.asarray(data_in['y'])

_x = np.ones((len(x1), 3))
_x[:, 1] = x1
_x[:, 2] = x2
_y = np.ones((len(x1), 2))
_y[:, 0] = 1 ^ y
_y[:, 1] = y

N = 2
K = 5
J = 2
v = np.ones((N+1, K))
w = np.ones((K+1, J))

__x = np.dot(_x, v)


F = np.reciprocal((1+np.exp(-__x)))

_F = np.ones((len(x1), K+1))
_F[:, 1:] = F
__F = np.dot(_F, w)
# __F = np.matmul(_F, w)

G = np.reciprocal(1+np.exp(-__F))

alpha1 = alpha2 = 0.001

for iterative in range(10):
    __x = np.dot(_x, v)
    F = np.reciprocal((1 + np.exp(-__x)))
    _F = np.ones((len(x1), K + 1))
    _F[:, 1:] = F
    __F = np.dot(_F, w)  # __F = np.matmul(_F, w)
    G = np.reciprocal(1 + np.exp(-__F))

    for i in range(len(x1)):
        for k in range(K+1):
            w[k] = w[k] - alpha1 * (G[i] - _y[i])*G[i]*(1 - G[i])*_F[i][k]

    for i in range(len(x1)):
        for j in range(J):
            for n in range(N+1):
                for k in range(1, K+1):
                    v[n][k-1] = v[n][k-1]-alpha2*(G[i][j]-_y[i][j])*G[i][j]*(1-G[i][j])*w[k][j]*F[i][k-1]*(1-F[i][k-1])*_x[i][n]

