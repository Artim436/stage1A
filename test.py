#!/usr/bin/env python3

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



n_matrices = 80 # how many SPD matrices to generate
n_dim = 2 # number of dimensions of the SPD matrices
sigma = 1.0  # dispersion of the Gaussian distribution
epsilon = 4.0  # parameter for controlling the distance between centers
random_state = 42  # ensure reproducibility

mean = np.eye(2)

sample_1 = sample_gaussian_spd(n_matrices, mean, sigma, random_state)



x = [sample_1[i][0][0] for i in range(n_matrices)]
y = [sample_1[i][1][1] for i in range(n_matrices)]
z = [sample_1[i][0][1] for i in range(n_matrices)]

i_start = 11

i_stop = 2

start = np.array([[x[i_start], z[i_start]],[z[i_start], y[i_start]]])

stop = np.array([[x[i_stop], z[i_stop]],[z[i_stop], y[i_stop]]])

def geodesic(t, start, stop):
    A = multi_dot([inv(sqrtm(start)), stop, inv(sqrtm(start))])
    B = matrix_power_homemade(A, t)
    C = multi_dot([sqrtm(start), B, sqrtm(start)])
    return C


def matrix_power_homemade(A, t):
    valp, vecp = eig(A)
    P = vecp
    D = np.diag(valp**t)
    return multi_dot([P, D, P.T])

phix = [geodesic(t/100, start, stop)[0][0] for t in range(100)]
phiy = [geodesic(t/100, start, stop)[1][1] for t in range(100)]
phiz = [geodesic(t/100, start, stop)[0][1] for t in range(100)]

ax = plt.axes(projection='3d')
ax.scatter3D(start[0][0], start[1][1], start[0][1], c="red")
ax.scatter3D(stop[0][0], stop[1][1], stop[1][0], c="blue")
ax.plot3D(phix, phiy, phiz, c="red")
ax.scatter3D(x, y, z, c="green")
ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# for angle in range(0, 360):
#     ax.view_init(30, angle)  #gif
#     plt.draw()
#     plt.pause(0.001)
plt.show()