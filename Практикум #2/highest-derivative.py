import matplotlib.pyplot as plt
import numpy as np


def p1(x):
    return 1


def q1(x):
    return 0


def f1(x):
    return 3


def p3(x):
    return 2


def q3(x):
    return -x


def f3(x):
    return x * x


def sweep(p, q, f, a, b, ca1, ca2, n=20):
    res = []
    w = (b - a) / n
    alpha = [0] * (n + 1)
    alpha[1] = -(ca1[1] / (w * ca1[0] - ca1[1]))
    betta = [0] * (n + 1)
    betta[1] = ca1[2] / (ca1[0] - ca1[1] / w)
    x = [0] * (n + 1)
    x[0] = a
    x[n] = b
    y = [0] * (n + 1)
    xdraw = [0] * (n + 1)
    ydraw = [0] * (n + 1)
    for i in range(1, n):
        x[i] = x[i - 1] + w
        c1 = 1 / w ** 2 - p(x[i]) / (2 * w)
        c2 = 1 / w ** 2 + p(x[i]) / (2 * w)
        c3 = -2 / w ** 2 + q(x[i])
        alpha[i + 1] = -c2 / (c1 * alpha[i] + c3)
        betta[i + 1] = (f(x[i]) - c1 * betta[i]) / (c1 * alpha[i] + c3)
    y[n] = (ca2[1] * betta[n] + ca2[2] * w) / (ca2[1] * (1 - alpha[n]) + ca2[0] * w)
    for i in range(n, 0, -1):
        y[i - 1] = y[i] * alpha[i] + betta[i]
    for i in range(n + 1):
        xdraw[i] = x[i]
        ydraw[i] = y[i]
        res.append((x[i], y[i]))
    plt.scatter(xdraw, ydraw, c='red', label='n = ' + str(n))
    return res


question = input('Введите количество итераций: (для значений по умолчанию нажмите Enter)\n')
n = 10
if question:
    n = int(question)
res = sweep(p1, q1, f1, 0, 1, [1, 0, 0], [0, 1, 3], 20)
length = len(res)
for i in range(length):
    print((res[i][0], res[i][1]), end="")
    if i < length - 1:
        print(', ', end="")
print()
x = np.linspace(0, 1)
y = 3 * x
plt.plot(x, y, c='blue', label='y(x)')
plt.legend(fontsize=12)
plt.grid(which='major')
plt.show()


res = sweep(p3, q3, f3, 0.6, 0.9, [0, 1, 0.7], [1, -0.5, 1], n)
length = len(res)
for i in range(length):
    print((res[i][0], res[i][1]), end="")
    if i < length - 1:
        print(', ', end="")
print()
plt.legend(fontsize=12)
plt.grid(which='major')
plt.show()