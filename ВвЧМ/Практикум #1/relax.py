import numpy as np


def condition(matrix: np.array):
    return np.linalg.cond(matrix)


def open_and_read(file_path):
    file = open(file_path, 'r')
    matrix = []
    b = []
    for i in file.readlines():
        str_inp = i.split()
        matrix.append(list(map(lambda x: int(x), str_inp[:-1])))
        b.append(int(str_inp[-1]))
    file.close()
    return matrix, b


def relax_method(matrix, b, w, ig, eps):
    ans = ig[:]
    length = len(matrix[0])
    iterations = 0
    residual = np.linalg.norm(np.matmul(matrix, ans) - b)
    while residual > eps:
        for i in range(length):
            sigma = 0
            for j in range(length):
                if j != i:
                    sigma += matrix[i][j] * ans[j]
            ans[i] = (1 - w) * ans[i] + (w / matrix[i][i]) * (b[i] - sigma)
        residual = np.linalg.norm(np.matmul(matrix, ans) - b)
        iterations += 1
    return iterations, ans


file_path = input('Путь до файла, в котором представлена система уравнений: (для значений по умолчанию нажмите Enter)\n')
if file_path:
    matrix, b = open_and_read(file_path)
else:
    for i in range(3):
        print(f'\n==========ФАЙЛ №{i + 1}==========\n')
        matrix, b = open_and_read(f'relax{i + 1}.txt')
        print('Число обусловленности матрицы A:', condition(np.array(matrix, dtype=float)))
        for j in range(3):
            iterations, phi = relax_method(np.array(matrix, dtype=float), np.array(b, dtype=float), 0.5 + 0.6 * j,
                                           np.zeros(4), 1e-8)
            print(f'За {iterations} итераци' + 'ю' * (
                    (iterations % 10 == 1) and ((iterations % 100 > 20) or (iterations % 100 < 5))) + 'и' * (
                          (5 > iterations % 10 > 1) and ((iterations % 100 > 20) or (iterations % 100 < 5))) + 'й' * (
                          (iterations % 10 > 4) or (not iterations % 10) or (
                          4 < iterations % 100 < 21)) + ' решения СЛАУ:', *phi)
        print(f'\n===========================\n')
