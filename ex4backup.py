import random
import numpy as np


def bp(xx, vx, vf, f0, g):
    y_ = np.empty(10)
    for k in range(g):
        for t in range(1, 9):
            f0[k][t] = f0[k][t - 1] * vf + xx[k][t - 1]*vx
        y_[k] = f0[k][8]
    return y_


x = np.empty((30, 8))
y = np.empty(30)
v_f = 1
v_x = 1
v_f_rp = random.random()
v_x_rp = random.random()
f = np.zeros((30, 9))
y_hat = np.empty(30)
E_1 = 1
E_0 = 4
diff = 1
delx = 0.001
for i in range(30):
    for q in range(8):
        x[i][q] = random.randrange(0, 2)
    y[i] = sum(x[i])


alpha1 = alpha2 = 0.01

while abs(E_1 - E_0) > diff:
    E_0 = E_1
    # FORWARD
    for i in range(30):
        for t in range(1, 9):
            f[i][t] = f[i][t-1]*v_f + x[i][t-1]*v_x
        y_hat[i] = f[i][8]
        E_1 += 0.5*((y_hat[i]-y[i])**2)

    for t in range(1, 9):
        for i in range(8):
            # BACK
            v_x = v_x - alpha1*(y_hat[i]-y[i])*x[i][t-1]*(v_f**(8-t))
            v_f = v_f - alpha2*(y_hat[i]-y[i])*f[i][t-1]*(v_f**(8-t))

x_test = np.empty((10, 8))
y_test = np.empty(10)
for i in range(10):
    for q in range(8):
        x_test[i][q] = random.randrange(0, 2)
    y_test[i] = sum(x_test[i])
print(x_test)
print(y_test)
print(bp(x_test, v_x, v_f, f, 10))

