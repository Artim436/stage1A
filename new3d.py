#!/usr/bin/env python3

from matplotlib import animation
import numpy as  np
from mpl_toolkits.mplot3d import Axes3D
from pyriemann.datasets import sample_gaussian_spd, generate_random_spd_matrix
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np
from scipy.linalg import sqrtm
from numpy.linalg import inv
from numpy.linalg import matrix_power
from numpy.linalg import multi_dot
from numpy.linalg import eig
from matplotlib import animation

n = 100

fig = plt.figure(figsize=(10,10))


ax =  Axes3D(fig)


for l in range(0,10):
    X = [(1+l)*t/n for t in range(n+1)]
    Y = [(1+l)-x for x in X]
    Z = [np.sqrt(X[k]*Y[k]) for k in range(n+1)]
    ax.plot3D(X, Y, Z, c="red")



def rotate(angle):
     ax.view_init(azim=angle)

angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save('inhadr_tsne1.gif', writer=animation.PillowWriter(fps=20))

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_zlim(0,8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()