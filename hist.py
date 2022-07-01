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
from scipy.stats import norm


n_matrices = 10000 # how many SPD matrices to generate
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
        theta = rd.random()*2*np.pi
        P = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        yield multi_dot([P, np.diag(np.array([l1,l2])), P.T])

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
        theta = rd.random()*2*np.pi
        P = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        yield multi_dot([P, np.diag(np.array([l1,l2])), P.T])

sample_1 = sample_gaussian_spd(n_matrices, mean, sigma)


H = []
HH = []


t11 = time.time()  
for cour in sample_1:
    d = distance_riemann(cour, np.eye(2))
    if d<= dmax:
        H.append(d)
t12 = time.time()
print("\n")
print(f'iter : {t12-t11}')
print("\n")

mu, std = norm.fit(np.asarray(H))

X = [(i*dmax/precision) for i in range(precision)]
plt.hist(H, bins= X, normed=True)
plt.ylim(0,1)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit Values: mu {:.2f} and std {:.2f}".format(mu, std)
plt.title(title)
plt.show()
