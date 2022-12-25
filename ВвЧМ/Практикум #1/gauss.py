import numpy as np


def condition(matrix: np.array):
    return np.linalg.cond(matrix)


def inverse_matrix(matrix: np.array):
    return np.linalg.inv(matrix)


def determinant(matrix: np.array):
    return np.linalg.det(matrix)


def gauss_method_with_max(matrix: np.array, b: np.array):
    length = len(matrix[0])
    for i in range(length):
        global_max = abs(matrix[i][i])
        index_row = i
        for j in range(i + 1, length):
            abs_value = abs(matrix[j][i])
            if abs_value > global_max:
                global_max = abs_value
                index_row = j
        for j in range(i, length):
            h = matrix[i][j]
            matrix[i][j] = matrix[index_row][j]
            matrix[index_row][j] = h
        u = b[i]
        b[i] = b[index_row]
        b[index_row] = u
        for j in range(i + 1, length):
            g = matrix[j][i]
            matrix[j] *= matrix[i][i]
            b[j] *= matrix[i][i]
            matrix[j] -= (matrix[i] * g)
            b[j] -= (b[i] * g)
            matrix[j] /= matrix[i][i]
            b[j] /= matrix[i][i]
    return solutions(matrix, b)


def gauss_method_with_zero(matrix: np.array, b: np.array):
    length = len(matrix[0])
    for i in range(length):
        flag = True
        if not matrix[i][i]:
            flag = False
            for j in range(i + 1, length):
                if matrix[j][i]:
                    h = matrix[i]
                    matrix[i] = matrix[j]
                    matrix[j] = h
                    flag = True
                    break
        if flag:
            for j in range(i + 1, length):
                g = matrix[j][i]
                matrix[j] *= matrix[i][i]
                b[j] *= matrix[i][i]
                matrix[j] -= (matrix[i] * g)
                b[j] -= (b[i] * g)
                matrix[j] /= matrix[i][i]
                b[j] /= matrix[i][i]
        else:
            return ValueError
    return solutions(matrix, b)


def solutions(matrix: np.array, b: np.array):
    length = len(matrix[0])
    ans = np.array([0] * length, dtype=float)
    for i in range(length - 1, -1, -1):
        summa = b[i]
        for j in range(i + 1, length):
            summa -= ans[j] * matrix[i][j]
        ans[i] = summa / matrix[i][i]
    return list(ans)


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


def testing(matrix, b):
    matrix2 = []
    b2 = []
    matrix2.extend(matrix)
    b2.extend(b)
    matrix2 = np.array(matrix2, dtype=float)
    b2 = np.array(b2, dtype=float)
    if determinant(matrix2):
        print('Метод Гаусса:', *gauss_method_with_zero(matrix2, b2))
        matrix2 = []
        b2 = []
        matrix2.extend(matrix)
        b2.extend(b)
        matrix2 = np.array(matrix2, dtype=float)
        b2 = np.array(b2, dtype=float)
        print('Метод Гаусса с выбором главного элемента:', *gauss_method_with_max(matrix2, b2))
        matrix2 = []
        matrix2.extend(matrix)
        matrix2 = np.array(matrix2, dtype=float)
    else:
        print('Система неразрешима')
    print('Определитель A:', determinant(matrix2))
    print('Обратная матрица к A:')
    print(inverse_matrix(matrix2))
    print('Число обусловленности матрицы A:', condition(matrix2))


file_path = input('Путь до файла, в котором представлена система уравнений: (для значений по умолчанию нажмите Enter)')
if file_path:
    matrix, b = open_and_read(file_path)
    testing(matrix, b)
else:
    for i in range(3):
        print(f'\n==========ФАЙЛ №{i + 1}==========\n')
        matrix, b = open_and_read(f'sys{i + 1}.txt')
        testing(matrix, b)
        print(f'\n===========================\n')
question = input('Хотите ввести свои n и m? (Y|N)\n')
if question == 'N':
    n = 30
    m = 20
else:
    n = int(input('n = '))
    m = int(input('m = '))
a = []
a_b = []
for i in range(n):
    a.append([])
    for j in range(n):
        elem = (i + j) / (m + n)
        if i == j:
            elem = n + m * m + j / m + i / n
        a[i].append(elem)
    a_b.append(m * i + n)
testing(a, a_b)
