#!/usr/bin/env python3


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


ax = Axes3D

# Make data.
X = np.arange(0, 10, 0.25)
Y = np.arange(0, 10, 0.25)
X, Y = np.meshgrid(X, Y)
Z = np.sqrt(X*Y)

# Plot the surface.
fig = plt.figure()
ax = Axes3D(fig)

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_xlim(0,10)
ax.set_ylim(0, 10)
ax.set_zlim(0,10)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5, label='z')

plt.show()