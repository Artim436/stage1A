#!/usr/bin/env python3
from functools import partial
import warnings
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm
from sklearn.utils import check_random_state
from joblib import Parallel, delayed
import math
import permutation
import matplotlib.pyplot as plt

def _pdf_r(r, sigma):
    """Pdf for the log of eigenvalues of a SPD matrix.

    Probability density function for the logarithm of the eigenvalues of a SPD
    matrix samples from the Riemannian Gaussian distribution. See Said et al.
    "Riemannian Gaussian distributions on the space of symmetric positive
    definite matrices" (2017) for the mathematical details.

    Parameters
    ----------
    r : ndarray, shape (n_dim,)
        Vector with the logarithm of the eigenvalues of a SPD matrix.
    sigma : float
        Dispersion of the Riemannian Gaussian distribution.

    Returns
    -------
    p : float
        Probability density function applied to data point r.
    """

    if (sigma <= 0):
        raise ValueError(f'sigma must be a positive number (Got {sigma})')

    n_dim = len(r)
    partial_1 = -np.sum(r**2) / (2 * sigma**2)
    partial_2 = 0
    for i in range(n_dim):
        for j in range(i + 1, n_dim):
            partial_2 = partial_2 + np.log(np.sinh(np.abs(r[i] - r[j]) / 2))

    return np.exp(partial_1 + partial_2)


def _rejection_sampling_side(n_dim, sigma, r_sample):
    """Auxiliary function for the 2D rejection sampling algorithm.

    It is used in the case where r is sampled with the function g+.

    Parameters
    ----------
    sigma : float
        Dispersion of the Riemannian Gaussian distribution.
    r_samples : ndarray, shape (1, n_dim)
        Sample of the r parameters of the Riemannian Gaussian distribution.

    Returns
    -------
    p : float
        Probability of acceptation.
    """
    mu = np.array([((n_dim+1-2*i)*sigma**2)/2 for i in range(1, n_dim+1)])
    cov_matrix = (sigma**2)*np.eye(n_dim)
    m = ((sigma)**n_dim)*(2*np.pi)**(n_dim/2)
    m = m/2**(n_dim*(n_dim-1)/2)
    s = 0
    for i in range(1, n_dim+1):
        s += ((n_dim+1-2*i)*sigma)**2
    m *= np.exp(s/8)
    if np.array_equal(r_sample, np.sort(r_sample)[::-1]):
        num = _pdf_r(r_sample, sigma)
        den = multivariate_normal.pdf(r_sample, mean=mu, cov=cov_matrix)*m
        return num / den
    return 0


def _rejection_sampling_nD(n_samples, n_dim, sigma, random_state=None, acceptation_probability=False):
    """Rejection sampling algorithm for the 2D case.

    Implementation of a rejection sampling algorithm. The implementation
    follows the description given in page 528 of Christopher Bishop's book
    "Pattern recognition and Machine Learning" (2006).

    Parameters
    ----------
    n_samples : int
        Number of samples to get from the target distribution.
    sigma : float
        Dispersion of the Riemannian Gaussian distribution.
    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    r_samples : ndarray, shape (n_samples, n_dim)
        Samples of the r parameters of the Riemannian Gaussian distribution.
    """
    mu = np.array([((n_dim+1-2*i)*sigma**2)/2 for i in range(1, n_dim+1)])
    cov_matrix = (sigma**2)*np.eye(n_dim)
    RES = []
    rs = check_random_state(random_state)
    if acceptation_probability == False:
        while cpt != n_samples:
            r_sample = multivariate_normal.rvs(mu, cov_matrix, 1, rs)
            res = _rejection_sampling_side(n_dim, sigma, r_sample)
            if rs.rand(1) < res:
                p = rs.randint(math.factorial(n_dim))
                RES.append(permutation.ranger(r_sample, p))
        return np.array(RES)
    cpt, cpt_2 = 0, 0
    while cpt != n_samples:
        cpt_2 +=1 
        r_sample = multivariate_normal.rvs(mu, cov_matrix, 1, rs)
        res = _rejection_sampling_side(n_dim, sigma, r_sample)
        if rs.rand(1) < res:
            p = rs.randint(math.factorial(n_dim))
            RES.append(permutation.ranger(r_sample, p))
            cpt += 1
    return np.array(RES), cpt/cpt_2



SIGMA = [1, 2, 3, 5, 10]

N_SAMPLES = [1, 10 ,100, 500, 1000, 2000, 5000, 10000]

N_DIM = [2, 3, 4, 5]



def plot_ndim(sigma=1, n_samples=1000):
    Y = []
    for n_dim in N_DIM:
        Y.append(_rejection_sampling_nD(n_samples=n_samples, n_dim=n_dim, sigma=sigma, acceptation_probability=True)[1])
    plt.plot(N_DIM, Y)
    plt.xlabel("dimension size")
    plt.ylabel("acceptation probability")
    plt.title(f"accepation probability vs dimension size for sigma={sigma} and n_samples={n_samples} ")
    plt.legend()
    plt.show()

def plot_sigma(n_dim=2, n_samples=1000):
    Y = []
    for sigma in SIGMA:
        Y.append(_rejection_sampling_nD(n_samples=n_samples, n_dim=n_dim, sigma=sigma, acceptation_probability=True)[1])
    plt.plot(SIGMA, Y)
    plt.xlabel("sigma")
    plt.ylabel("acceptation probability")
    plt.title(f"accepation probability vs sigma for n_dim={n_dim} and n_samples={n_samples} ")
    plt.legend()
    plt.show()

def plot_nsamples(n_dim=2, sigma=1):
    Y = []
    for n_samples in N_SAMPLES:
        Y.append(_rejection_sampling_nD(n_samples=n_samples, n_dim=n_dim, sigma=sigma, acceptation_probability=True)[1])
    plt.plot(N_SAMPLES, Y)
    plt.xlabel("number of samples")
    plt.ylabel("acceptation probability")
    plt.title(f"accepation probability vs number of sample for sigma={sigma} and n_samples={n_samples} ")
    plt.legend()
    plt.show() 



for sigma in SIGMA:
    plot_ndim(sigma=sigma, n_samples=100)






#### test pour voir si les eigenvals ont une distribution log normal


# sigma = 3
# n_dim = 5
# n_sample = 1000
# R_4, p = _rejection_sampling_nD(n_sample, n_dim, sigma)
# print(p)
# H =[]
# for k in range(n_sample):
#     r = R_4[k]
#     H.append(np.sum(r))


# plt.hist(H, bins=100, normed=True)

# xmin, xmax = plt.xlim()
# print(xmin, xmax)
# x = np.linspace(xmin, xmax, 100)
# mu, std = norm.fit(np.array(H))
# pdf = norm.pdf(x, mu, std)
# plt.plot(x, pdf, 'k', linewidth=2)

# title = "Fit Values: mu {:.2f} and std {:.2f} = root({:.2f} x {:.2f}**2)".format(mu, std, n_dim, sigma)
# plt.title(title)
# plt.show()


