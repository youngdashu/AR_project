#!/usr/bin/env python

import numpy as np
# from mpi4py import MPI
import matplotlib.pyplot as plt
from tqdm import trange

# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()

N = 1000
n_iterations = 500
matrix = np.ones(N)
matrix[0] = 0
matrix[-1] = 0
planck =  6.626 * 10e-2 # ×10−34 joule-hertz−1
m = 1
n = 3
w = 5

xs = np.linspace(-5, 5, N)
h = xs[1] - xs[0]

def p(i):
    return ((2 * m * (n + 1 / 2) * w) - (m ** 2 * w ** 2 * xs[i] ** 2) / planck) * h ** 2 - 2
    # return (3 - x ** 2) * h ** - 2

def calculate_xi(i):
    if i == 0:
        return p(i) * matrix[i] + matrix[i + 1]
    elif 0 < i < N - 1:
        res = -p(i) * matrix[i] + matrix[i + 1] + matrix[i - 1]
        # print(res)
        return res
    elif i == N - 1:
        return  p(i) * matrix[i] + matrix[i - 1]
    else:
        raise ValueError("Parameter i is out of bounds")





print("START")
result = None
for _ in trange(n_iterations):
    result = []
    # if size == 1:
    lower_border = 0
    upper_border = N
    # else:
    #     lower_border = (n // (size - 1)) * (rank - 1)
    #     upper_border = (n // (size - 1)) * rank if rank != size - 1 else (n // (size - 1)) * rank + n % (size - 1)

    for i in range(lower_border, upper_border):
        result.append(calculate_xi(i))

    matrix = result

    # data = comm.gather(result, root=0)
    # if rank == 0:
    #     if size != 1:
    #         data.pop(0)
    #     new_matrix = np.concatenate(data)
    #     matrix = comm.bcast(new_matrix, root=0)
    # else:
    #     matrix = comm.bcast(None, root=0)

def plot_matrix(matrix, n_iterations):
    plt.plot(matrix)
    # plt.ylim(ymin=-0.25, ymax=0.25)
    plt.title(f'Quantum oscillator for N={N}, n_iterations = {n_iterations}')
    plt.show()

plot_matrix(result, n_iterations)
