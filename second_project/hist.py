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
from functools import partial
import warnings
import numpy as np
from sklearn.utils import check_random_state
from joblib import Parallel, delayed
from module_hist import _sample_parameter_r
from scipy.stats import gengamma
from scipy.stats import chi2
from numpy.random import chisquare
from scipy.special import gamma


n_matrices = 10000 # how many SPD matrices to generate
n_dim = 2 # number of dimensions of the SPD matrices
sigma = 1.0  # dispersion of the Gaussian distribution
random_state = 42  # ensure reproducibility
precision = 100
dmax = 5

mean = np.eye(2)


def sample_norm_fct_iter():
    for i in range(n_matrices):
        l1 = np.random.normal(0,1)
        l2 = np.random.normal(0,1)
        # theta = rd.random()*2*np.pi
        # P = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        yield np.diag(np.array([np.exp(l1),np.exp(l2)]))

def pdf_nakagami(x, m, o):
    n = m**m
    n *= 2
    n *= 1/(gamma(m)*(o**m))
    n *= x**(2*m-1)
    n *= np.exp(-m*x**2/o)
    return n

Hnorm = []
Hnorm2 = []

for cour in sample_norm_fct_iter():
    d = (distance_riemann(cour, np.eye(2)))
    if d<= dmax:
        Hnorm.append(d**2)

for cour in sample_norm_fct_iter():
    d = (distance_riemann(cour, np.eye(2)))
    if d<= dmax:
        Hnorm2.append(d)


muchi = 2 #chi2.fit(np.asarray(Hnorm))[0]

print(chi2.fit(np.asarray(Hnorm)))

X = [(i*dmax/precision) for i in range(precision)]

percs = np.linspace(1, 100, 100)

shape = 2


plt.subplot(221)
plt.hist(Hnorm, bins= X, normed=True)
plt.ylim(0,1.5)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
pnorm = chi2.pdf(x, 2)
plt.plot(x, pnorm, 'k', linewidth=2)
title = "Fit Values: shape = {:.2f} for Chi2's law".format(muchi)
plt.title(title)

plt.subplot(222)
qgauss = np.percentile(np.random.chisquare(muchi, n_matrices), percs)
qnorm = np.percentile(np.asarray(Hnorm), percs)
plt.plot(qgauss, qnorm, ls="", marker="o")
x = np.linspace(np.min((qgauss.min(),qnorm.min())), np.max((qgauss.max(),qnorm.max())))
plt.plot(x,x, color="k", ls="--")

m = 1
o = 2
plt.subplot(223)
plt.hist(Hnorm2, bins= X, normed=True)
plt.ylim(0,1.5)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
puni = [pdf_nakagami(x, 1, 2) for x in x]
plt.plot(x, puni, 'k', linewidth=2)
title = "Fit Values: m = {:.2f} and Omega = {:.2f} for Nakagami's law".format(m, o)
plt.title(title)

plt.tight_layout()
plt.show()
