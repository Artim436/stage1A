#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import random as rd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils import check_random_state
from scipy.stats import multivariate_normal
import time

start1 = time.time()
def rejection_sampling_A(sigma, r):
    MU_A = np.array([sigma**2/2, -sigma**2/2])
    COV_MATRIX = (sigma**2)*np.eye(2)
    M = np.pi*(sigma**2)*np.exp(sigma**2/4)
    if r[0] >= r[1]:
        M = np.pi*(sigma**2)*np.exp(sigma**2/4)
        num = np.exp(-1/(2*sigma**2) * np.sum(r**2)) * np.sinh((r[0] - r[1])/2) * 1/M
        den = multivariate_normal.pdf(r, mean=MU_A, cov=COV_MATRIX)
        return num / den
    return 0
    
def rejection_sampling_B(sigma, r):
    MU_B = np.array([-sigma**2/2, sigma**2/2])
    COV_MATRIX = (sigma**2)*np.eye(2)
    M = np.pi*(sigma**2)*np.exp(sigma**2/4)
    if r[0] < r[1]:
        M = np.pi*(sigma**2)*np.exp(sigma**2/4)
        num = np.exp(-1/(2*sigma**2) * np.sum(r**2)) * np.sinh((r[1] - r[0])/2)
        den = multivariate_normal.pdf(r, mean=MU_B, cov=COV_MATRIX)*M
        return num/den
    return 0


def rejection_sampling(sigma, n):
    MU_A = np.array([sigma**2/2, -sigma**2/2])
    MU_B = np.array([-sigma**2/2, sigma**2/2])
    COV_MATRIX = (sigma**2)*np.eye(2)
    M = np.pi*(sigma**2)*np.exp(sigma**2/4)
    RES = []
    cpt = 0
    while cpt != n:
        if np.random.binomial(1, 0.5, 1) == 1:
            r = multivariate_normal.rvs(MU_A, COV_MATRIX, 1)
            res = rejection_sampling_A(sigma, r)
            if np.random.rand(1) < res:
                RES.append(r)
                cpt += 1
        else:
            r = multivariate_normal.rvs(MU_B, COV_MATRIX, 1)
            res = rejection_sampling_B(sigma, r)
            if np.random.rand(1) < res:
                RES.append(r)
                cpt+=1
    return np.array(RES)

R = rejection_sampling(1, 10000)
end1 = time.time()
print(np.shape(R))

start2 = time.time()
def rejection_sampling_A(sigma, r):
    MU_A = np.array([sigma**2/2, -sigma**2/2])
    COV_MATRIX = (sigma**2)*np.eye(2)
    M = np.pi*(sigma**2)*np.exp(sigma**2/4)
    if r[0] >= r[1]:
        M = np.pi*(sigma**2)*np.exp(sigma**2/4)
        num = np.exp(-1/(2*sigma**2) * np.sum(r**2)) * np.sinh((r[0] - r[1])/2) * 1/M
        den = multivariate_normal.pdf(r, mean=MU_A, cov=COV_MATRIX)
        return num / den
    return 0
    
def rejection_sampling_B(sigma, r):
    MU_B = np.array([-sigma**2/2, sigma**2/2])
    COV_MATRIX = (sigma**2)*np.eye(2)
    M = np.pi*(sigma**2)*np.exp(sigma**2/4)
    if r[0] < r[1]:
        M = np.pi*(sigma**2)*np.exp(sigma**2/4)
        num = np.exp(-1/(2*sigma**2) * np.sum(r**2)) * np.sinh((r[1] - r[0])/2)
        den = multivariate_normal.pdf(r, mean=MU_B, cov=COV_MATRIX)*M
        return num/den
    return 0


def rejection_sampling(sigma, n):
    MU_A = np.array([sigma**2/2, -sigma**2/2])
    MU_B = np.array([-sigma**2/2, sigma**2/2])
    COV_MATRIX = (sigma**2)*np.eye(2)
    M = np.pi*(sigma**2)*np.exp(sigma**2/4)
    RES = []
    cpt = 0
    B = np.random.binomial(1, 0.5, 2*n)
    M1 = multivariate_normal.rvs(MU_A, COV_MATRIX, 2*n)
    M2 = multivariate_normal.rvs(MU_B, COV_MATRIX, 2*n)
    N = np.random.rand(2*n)
    i = 0
    k = 0
    l = 0
    while cpt != n:
        if B[i] == 1:
            r = M1[k]
            k+=1
            res = rejection_sampling_A(sigma, r)
            if N[i] < res:
                RES.append(r)
                cpt += 1
            i+=1
        else:
            r = M2[l]
            l+=1
            res = rejection_sampling_B(sigma, r)
            if N[i] < res:
                RES.append(r)
                cpt+=1
            i+=1
    return np.array(RES)

R = rejection_sampling(1, 10000)
end2 = time.time()

print(np.shape(R))
print(end1 - start1)
print(end2 - start2)

sigma = 1

T = 5
rx = np.linspace(-T, T, 200)
ry = np.linspace(-T, T, 200)
RX, RY = np.meshgrid(rx, ry)

F = np.exp(-1/(2*sigma**2)*(RX**2 + RY**2)) * np.sinh(np.abs(RX - RY)/2)

fig, ax = plt.subplots(figsize=(12, 6), ncols=2)
ax[0].contour(RX, RY, F)
ax[0].scatter(R[:,0],R[:,1], s=2, alpha=0.5)
ax[1].hexbin(R[:,0], R[:,1], extent=(-T, T, -T, T))
plt.show()