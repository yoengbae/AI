import random
import numpy as np


def sign(h):
    if h > 0:
        return 1
    else:
        return 0


def bp_train(x, y, f_):
    E_1 = 1
    E_0 = 4
    v_f = 1.0
    v_x = 1.0
    y_hat = np.empty(30)
    alpha1 = alpha2 = 0.01
    while abs(E_1 - E_0) > 1:
        E_0 = E_1
        for i in range(30):
            for t in range(1, 9):
                f[i][t] = f_[i][t - 1] * v_f + x[i][t - 1] * v_x
            y_hat[i] = f_[i][8]
            E_1 += 0.5 * ((y_hat[i] - y[i]) ** 2)

        for t in range(1, 9):
            for i in range(30):
                v_x = v_x - alpha1 * (y_hat[i] - y[i]) * x[i][t - 1] * (v_f ** (8 - t))
                v_f = v_f - alpha2 * (y_hat[i] - y[i]) * f_[i][t - 1] * (v_f ** (8 - t))
    return v_x, v_f


def predict(xx, vx, vf, f0, g):
    y_ = np.empty(10)
    for k in range(g):
        for t in range(1, 9):
            f0[k][t] = f0[k][t - 1] * vf + xx[k][t - 1]*vx
        y_[k] = f0[k][8]
    return y_


def rp_train(x, y, f_):
    E_1 = 1
    E_0 = 4
    v_f2 = 10
    v_x2 = 1
    dx = df = 0.001
    tmp1 = tmp2 = 1
    y_hat = np.empty(30)
    while abs(E_1 - E_0) > 1:
        E_0 = E_1
        dif_e_vx = dif_e_vf = 0
        for i in range(30):
            for t in range(1, 9):
                f[i][t] = f_[i][t - 1] * v_f2 + x[i][t - 1] * v_x2
            y_hat[i] = f_[i][8]
            E_1 += 0.5 * ((y_hat[i] - y[i]) ** 2)

        for t in range(1, 9):
            for i in range(8):
                dif_e_vx += (y_hat[i] - y[i]) * x[i][t - 1] * (v_f2 ** (8 - t))
                dif_e_vf += (y_hat[i] - y[i]) * f_[i][t - 1] * (v_f2 ** (8 - t))
        if sign(dif_e_vx) == sign(tmp1):
            dx = dx*1.2
        else:
            dx = dx*0.5
        if sign(dif_e_vf) == sign(tmp2):
            df = df*1.2
        else:
            df = df*0.5
        tmp1 = dif_e_vx
        tmp2 = dif_e_vf
        if dif_e_vx > 0:
            v_x2 = v_x2 - dx
        elif dif_e_vx < 0:
            v_x2 = v_x2 + dx

        if dif_e_vf > 0:
            v_f2 = v_f2 - df
        elif dif_e_vf < 0:
            v_f2 = v_f2 + df
    return v_x2, v_f2


# def gc_train():


x_train = np.empty((30, 8))
y_train = np.empty(30)
f = np.zeros((30, 9))
for i in range(30):
    for q in range(8):
        x_train[i][q] = random.randrange(0, 2)
    y_train[i] = sum(x_train[i])

v_x_bp, v_f_bp = bp_train(x_train, y_train, f)
v_x_rp, v_f_rp = rp_train(x_train, y_train, f)

x_test = np.empty((10, 8))
y_test = np.empty(10)
for i in range(10):
    for q in range(8):
        x_test[i][q] = random.randrange(0, 2)
    y_test[i] = sum(x_test[i])

print(x_test)
print(y_test)
print(predict(x_test, v_x_bp, v_f_bp, f, 10))
print(predict(x_test, v_x_rp, v_f_rp, f, 10))
