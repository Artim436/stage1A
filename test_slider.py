#!/usr/bin/env python3

import numpy as  np
from mpl_toolkits.mplot3d import axes3d
from pyriemann.datasets import sample_gaussian_spd, generate_random_spd_matrix
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


n_matrices = 10000  # how many SPD matrices to generate


def f(k, sigma):
    n_dim = 2 # number of dimensions of the SPD matrices
    random_state = 42  # ensure reproducibility
    mean = np.eye(n_dim)
    sample_1 = sample_gaussian_spd(n_matrices, mean, sigma, random_state)
    x = [sample_1[i][0][0] for i in range(n_matrices)]
    y = [sample_1[i][1][1] for i in range(n_matrices)]
    z = [sample_1[i][0][1] for i in range(n_matrices)]
    return x[k]


k = np.linspace(0, n_matrices, n_matrices)

init_sigma = 1.0

fig, ax = plt.subplots()
line, = plt.plot(k, f(k, init_sigma), lw=2)
plt.subplots_adjust(left=0.25, bottom=0.25)
axsigma = plt.axes([1.0, 1.25, 1.50, 1.75, 2.0])
sigma_slider = Slider(ax = axsigma, label="sigma", valmin=0, valmax=2, valinit=init_sigma)


def update(val):
    line.set_ydata(f(k, sigma_slider.val))
    fig.canvas.draw_idle()



sigma_slider.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color='red', hovercolor='0.975')

def reset(event):
    sigma_slider.reset()
button.on_clicked(reset)
plt.show()
