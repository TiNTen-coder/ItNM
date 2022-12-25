from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from math import *


def f1(x, y):
    return (x - x * x) * y


def g1(x, u, v):
    return cos(x + 1.5 * v) - u


def g2(x, u, v):
    return -(v * v) + 2.3 * u - 1.2


def runge_kutta_double(f, w, n, x0, y0):
    res = [[x0, y0]]
    x = [x0]
    y = [y0]
    w = w / n
    for i in range(n):
        y0 += (f(x0, y0) + f(x0 + w, y0 + w * f(x0, y0))) * w / 2
        x0 += w
        x.append(x0)
        y.append(y0)
        res.append([x0, y0])
    plt.scatter(x, y, c='green', label='R-K2')
    return res


def runge_kutta_quaternary(f, w, n, x0, y0):
    res = [[x0, y0]]
    x = [x0]
    y = [y0]
    w = w / n
    for i in range(n):
        c1 = f(x0, y0)
        c2 = f(x0 + w / 2, y0 + w / 2 * c1)
        c3 = f(x0 + w / 2, y0 + w / 2 * c2)
        c4 = f(x0 + w, y0 + w * c3)
        x0 += w
        y0 += w / 6 * (c1 + 2 * c2 + 2 * c3 + c4)
        x.append(x0)
        y.append(y0)
        res.append([x0, y0])
    plt.scatter(x, y, c='red', label='R-K4')
    return res


def runge_kutta_double_for_sys(funcs, w, n, x0, y0):
    w = w / n
    length = len(funcs)
    res = [[x0, y0]]
    x = [x0]
    y1 = [y0[0]]
    y2 = [y0[1]]
    for i in range(n):
        t1 = [0] * length
        t2 = [0] * length
        for j in range(length):
            t1[j] = funcs[j](x0, y0[0], y0[1])
            t2[j] = funcs[j](x0 + w, y0[0] + w * t1[j], y0[1] + w * t1[j])
            y0[j] += (t1[j] + t2[j]) * w / 2
        x0 += w
        x.append(x0)
        y1.append(y0[0])
        y2.append(y0[1])
        res.append([x0, deepcopy(y0)])
    plt.scatter(x, y1, c='green', label='u2')
    plt.scatter(x, y2, c='red', label='v2')
    return res


def runge_kutta_quaternary_for_sys(funcs, w, n, x0, y0):
    w = w / n
    length = len(funcs)
    res = [[x0, y0]]
    x = [x0]
    y1 = [y0[0]]
    y2 = [y0[1]]
    for i in range(n):
        c1 = [0] * length
        c2 = [0] * length
        c3 = [0] * length
        c4 = [0] * length
        c1[0] = funcs[0](x0, y0[0], y0[1])
        c1[1] = funcs[1](x0, y0[0], y0[1])
        c2[0] = funcs[0](x0 + w / 2, y0[0] + w / 2 * c1[0], y0[1] + w / 2 * c1[1])
        c2[1] = funcs[1](x0 + w / 2, y0[0] + w / 2 * c1[0], y0[1] + w / 2 * c1[1])
        c3[0] = funcs[0](x0 + w / 2, y0[0] + w / 2 * c2[0], y0[1] + w / 2 * c2[1])
        c3[1] = funcs[1](x0 + w / 2, y0[0] + w / 2 * c2[0], y0[1] + w / 2 * c2[1])
        c4[0] = funcs[0](x0 + w, y0[0] + w * c3[0], y0[1] + w * c3[1])
        c4[1] = funcs[1](x0 + w, y0[0] + w * c3[0], y0[1] + w * c3[1])
        y0[0] += w / 6 * (c1[0] + 2 * c2[0] + 2 * c3[0] + c4[0])
        y0[1] += w / 6 * (c1[1] + 2 * c2[1] + 2 * c3[1] + c4[1])
        for j in range(2, length):
            c1[j] = funcs[j](x0, y0[0], y0[1])
            c2[j] = funcs[j](x0 + w / 2, y0[0] + w / 2 * c1[0], y0[1] + w / 2 * c1[1])
            c3[j] = funcs[j](x0 + w / 2, y0[0] + w / 2 * c2[0], y0[1] + w / 2 * c2[1])
            c4[j] = funcs[j](x0 + w, y0[0] + w * c3[0], y0[1] + w * c3[1])
            y0[j] += w / 6 * (c1[j] + 2 * c2[j] + 2 * c3[j] + c4[j])
        x0 += w
        x.append(x0)
        y1.append(y0[0])
        y2.append(y0[1])
        res.append([x0, deepcopy(y0)])
    plt.scatter(x, y1, c='magenta', label='u4')
    plt.scatter(x, y2, c='orange', label='v4')
    return res


n = 50
runge_kutta_double(f1, 1, n, *[0, 1])
runge_kutta_quaternary(f1, 1, n // 2, *[0, 1])
x = np.linspace(0, 1)
y = e ** ((-1 / 6) * x * x * (-3 + 2 * x))
plt.plot(x, y, c='blue', label='y(x)')
plt.legend(fontsize=12)
plt.grid(which='major')
plt.show()

n = 30
runge_kutta_double_for_sys([g1, g2], 3, n, 0, [0.25, 1])
runge_kutta_quaternary_for_sys([g1, g2], 3, n, 0, [0.25, 1])
plt.legend(fontsize=12)
plt.grid(which='major')
plt.show()
