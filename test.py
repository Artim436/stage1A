#!/usr/bin/env python3

import time
import numpy as  np
from mpl_toolkits.mplot3d import axes3d
from pyriemann.datasets import sample_gaussian_spd, generate_random_spd_matrix
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np
from scipy.linalg import sqrtm
from numpy.linalg import inv
from numpy.linalg import matrix_power
from numpy.linalg import multi_dot
from numpy.linalg import eig
from pyriemann.utils.distance import distance_riemann
import random as rd


def f(x):
    return (x*(1-x))**(1/2)

X  = [x/100 for x in range(0,101)]
Y = [f(x) for x in X]
plt.plot(X, Y)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("x")
plt.ylabel("y = sqrt(x(1-x))")
plt.legend()
plt.grid()
plt.show()


# n_matrices = 100000 # how many SPD matrices to generate
# n_dim = 2 # number of dimensions of the SPD matrices
# sigma = 1.0  # dispersion of the Gaussian distribution
# random_state = 42  # ensure reproducibility
# precision = 100
# dmax = 3

# mean = np.eye(2)



# def sample_norm_fct():
#     l1 = np.random.normal(1,1)
#     l2 = np.random.normal(1,1)
#     theta = rd.random()*2*np.pi
#     P = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
#     return multi_dot([P, np.diag(np.array([l1,l2])), P.T])

# HH = []


# t21 = time.time()
# sample_norm = [sample_norm_fct() for _ in range(n_matrices)]
# for M in sample_norm:
#     d = distance_riemann(M, np.eye(2))
#     if d<= dmax:
#         HH.append(d)
# t22 = time.time()
# print("\n")
# print(f'liste : {t22-t21}')
# print("\n")


# X = [(i*dmax/precision) for i in range(precision)]
# plt.hist(HH, bins= X, normed=True)
# plt.show()
