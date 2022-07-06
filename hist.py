#!/usr/bin/env python3

import time
import numpy as  np
from mpl_toolkits.mplot3d import axes3d
from pyriemann.datasets import sample_gaussian_spd, generate_random_spd_matrix
# import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np
from scipy.linalg import sqrtm
from numpy.linalg import inv
from numpy.linalg import matrix_power
from numpy.linalg import multi_dot
from numpy.linalg import eig
from pyriemann.utils.distance import distance_riemann
import random as rd
from scipy.stats import norm
import numpy as np
import statsmodels.api as sm
import pylab as plt


n_matrices = 1000 # how many SPD matrices to generate
n_dim = 2 # number of dimensions of the SPD matrices
sigma = 1.0  # dispersion of the Gaussian distribution
random_state = 42  # ensure reproducibility
precision = 100
dmax = 5

mean = np.eye(2)


def sample_norm_fct_iter():
    for i in range(n_matrices):
        l1 = np.random.normal(1,1)
        l2 = np.random.normal(1,1)
        # theta = rd.random()*2*np.pi
        # P = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        yield np.diag(np.array([l1,l2]))

def sample_norm_fct():
    l1 = np.random.normal(1,1)
    l2 = np.random.normal(1,1)
    theta = rd.random()*2*np.pi
    P = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    return multi_dot([P, np.diag(np.array([l1,l2])), P.T])

def sample_uni_fct_iter():
    for i in range(n_matrices):
        l1 = rd.random()
        l2 = rd.random()
        # theta = rd.random()*2*np.pi
        # P = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        yield np.diag(np.array([l1,l2]))

sample_1 = sample_gaussian_spd(n_matrices, mean, sigma)


Huni = []
Hnorm = []
Hgauss = []

for cour in sample_norm_fct_iter():
    d = (distance_riemann(cour, np.eye(2)))
    if d<= dmax:
        Hnorm.append(d)

for cour in sample_uni_fct_iter():
    d = (distance_riemann(cour, np.eye(2)))
    if d<= dmax:
        Huni.append(d)

for cour in sample_1:
    d = (distance_riemann(cour, np.eye(2)))
    if d<= dmax:
        Hgauss.append(d)

munorm, stdnorm = norm.fit(np.asarray(Hnorm))
muni, stduni = norm.fit(np.asarray(Huni))
mugauss, stdgauss = norm.fit(np.asarray(Hgauss))

X = [(i*dmax/precision) for i in range(precision)]

percs = np.linspace(1, 100, 100)
qgaussprac = np.percentile(np.asarray(Hgauss), percs)
qgauss = np.percentile(np.random.normal(1.16, 0.49, n_matrices), percs)



plt.subplot(321)
plt.hist(Hnorm, bins= X, normed=True)
plt.ylim(0,1)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
pnorm = norm.pdf(x, munorm, stdnorm)
plt.plot(x, pnorm, 'k', linewidth=2)
title = "Fit Values: munorm {:.2f} and stdnorm {:.2f}".format(munorm, stdnorm)
plt.title(title)

plt.subplot(322)
qnorm = np.percentile(np.asarray(Hnorm), percs)
plt.plot(qgauss, qnorm, ls="", marker="o")
x = np.linspace(np.min((qgauss.min(),qnorm.min())), np.max((qgauss.max(),qnorm.max())))
plt.plot(x,x, color="k", ls="--")


plt.subplot(323)
plt.hist(Huni, bins= X, normed=True)
plt.ylim(0,1)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
puni = norm.pdf(x, muni, stduni)
plt.plot(x, puni, 'k', linewidth=2)
title = "Fit Values: muni{:.2f} and stduni {:.2f}".format(muni, stduni)
plt.title(title)

plt.subplot(324)
quni = np.percentile(np.asarray(Huni), percs)
plt.plot(qgauss, quni, ls="", marker="o")
x = np.linspace(np.min((qgauss.min(),quni.min())), np.max((qgauss.max(),quni.max())))
plt.plot(x,x, color="k", ls="--")

plt.subplot(325)
plt.hist(Hgauss, bins= X, normed=True)
plt.ylim(0,1)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
pgauss = norm.pdf(x, mugauss, stdgauss)
plt.plot(x, pgauss, 'k', linewidth=2)
title = "Fit Values: mugauss{:.2f} and stdgauss {:.2f}".format(mugauss, stdgauss)
plt.title(title)

plt.subplot(326)
plt.plot(qgaussprac, qgauss, ls="", marker="o")
x = np.linspace(np.min((qgauss.min(),qgauss.min())), np.max((qgauss.max(),qgauss.max())))
plt.plot(x,x, color="k", ls="--")

plt.show()
