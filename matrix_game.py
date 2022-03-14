import numpy as np 
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import sys
from sys import argv

#Поиск седловых точек матрицы
def saddle_point(A) -> tuple:
    m = A.shape[0]
    n = A.shape[1]

    min_rows = np.min(A, axis = 1)
    max_cols = np.max(A, axis = 0)

    maxmin = np.max(min_rows)
    minmax = np.min(max_cols)

    rows_idx = None
    cols_idx = None
    if (maxmin == minmax):
        rows_idx = np.argmax(min_rows)
        cols_idx = np.argmin(max_cols)

    return (rows_idx, cols_idx)

#Поиск оптимальных стратегий
def nash_equilibrilium(A):
    #Проверяем, есть ли седловые точки
    prob_saddle_point = saddle_point(A)

    #Если есть, то все ок, дальше ничего не считаем
    if (prob_saddle_point != (None, None)):
        p = np.zeros(A.shape[0])
        q = np.zeros(A.shape[1])
        p[prob_saddle_point[0]] = 1
        q[prob_saddle_point[1]] = 1
        game_value = A[prob_saddle_point[0]][prob_saddle_point[1]]
        return p, q, game_value, "pure"

    #Если седловых точек нет и все не ок, разбиваем на две задачи линейного программирования и решаем симлекс-методом.
    #Спасибо scipy за существование linprog

    m = A.shape[0]
    n = A.shape[1]
    min_a = A.min()
    dobavochka = 0
    if (min_a < 0):
        dobavochka = 1 + abs(min_a)
    A = A + dobavochka
    z_1 = np.ones(m)
    z_2 = np.ones(n) * (-1)
    b_1 = np.ones(n) * (-1)
    b_2 = np.ones(m)
    A_1 = A.transpose() * (-1)
    A_2 = A

    player_1 = linprog(c = z_1, A_ub = A_1, b_ub = b_1, bounds = (0, float("inf")), method = "simplex")
    player_2 = linprog(c = z_2, A_ub = A_2, b_ub = b_2, bounds = (0, float("inf")), method = "simplex")

    game_value = 1 / player_1.fun
    p = player_1.x * game_value
    q = player_2.x * game_value
    return p, q, game_value - dobavochka, "mixed"

#Печать матрицы с округлением ее элементов до 3 знака после запятой
def print_matrix(A):
    m = A.shape[0]
    n = A.shape[1]
    for i in range(m):
        for j in range (n):
            print ("%.3f" % (a[i][j]), end = '  ')
        print()

#Печать вектора с округлением его элементов до 3 знака после запятой
def print_vector(v):
    n = v.shape[0]
    print ('[', end = ' ')
    for i in range(n):
        print ("%.3f" % (v[i]), end = '  ')
    print (']')

#Визуализация вектора оптимальной стратегии
def visualization(v):
    plt.clf()
    n = len(v)
    x = np.linspace(1, n, n)  
    plt.axis([0, n + 1, 0, max(v) + 1]) 
    plt.style.use('ggplot')
    plt.stem(x, v, basefmt=' ')
    plt.show()

#Чтение матрицы из файла
def input_from_file(filename):
    return np.loadtxt(filename, delimiter=' ')

#Печать оптимальных стратегий
def print_results(p, q, game_value, strategy_type):
    print("Solution in " + strategy_type + " strategies.")
    print("Game value is: ", "%.3f" % (game_value))
    print("First player's optimal strategy")
    print_vector(p)
    print("Second player's optimal strategy")
    print_vector(q)

def main(file_name):
    try:
        A = input_from_file(file_name)
    except Exception:
        print("Incorrect file")
        sys.exit()
    p, q, game_value, strategy_type = nash_equilibrilium(A)
    print_results(p, q, game_value, strategy_type)
    visualization(p)
    visualization(q)

s = argv[1]
main(s)
    

    
  

#a = np.array([[4, 0, 6, 2, 2, 1], [3, 8, 4, 10, 4, 4], [1, 2, 6, 5, 0, 0], [6, 6, 4, 4, 10, 3], [10, 4, 6, 4, 0, 9], [10, 7, 0, 7, 9, 8]])
#print(nash_equilibrilium(a))