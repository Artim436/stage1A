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


def _pdf_r(r, sigma):
    """Pdf for the log of eigenvalues of a SPD matrix.
    Probability deπnsity function for the logarithm of the eigenvalues of a SPD
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

def _slice_sampling(ptarget, n_samples, x0, n_burnin=20, thin=10,
                    random_state=None, n_jobs=1):
    """Slice sampling procedure.
    Implementation of a slice sampling algorithm for sampling from any target
    pdf or a multiple of it. The implementation follows the description given
    in page 375 of David McKay's book "Information Theory, Inference, and
    Learning Algorithms" (2003).
    Parameters
    ----------
    ptarget : function with one input
        The target pdf to sample from or a multiple of it.
    n_samples : int
        How many samples to get from the ptarget distribution.
    x0 : array
        Initial state for the MCMC procedure. Note that the shape of this array
        defines the dimensionality n_dim of the data points to be sampled.
    n_burnin : int, default=20
        How many samples to discard from the beginning of the chain generated
        by the slice sampling procedure. Usually the first samples are prone to
        non-stationary behavior and do not follow very well the target pdf.
    thin : int, default=10
        Thinning factor for the slice sampling procedure. MCMC samples are
        often correlated between them, so taking one sample every `thin`
        samples can help reducing this correlation. Note that this makes the
        algorithm actually sample `thin x n_samples` samples from the pdf, so
        expect the whole sampling procedure to take longer.
    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible output across multiple function calls.
    n_jobs : int, default=1
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel. If -1 all CPUs are used.
    Returns
    -------
    samples : ndarray, shape (n_samples, n_dim)
        Samples from the target pdf.
    """

    if (n_samples <= 0) or (not isinstance(n_samples, int)):
        raise ValueError(
            f'n_samples must be a positive integer (Got {n_samples})')
    if (n_burnin <= 0) or (not isinstance(n_burnin, int)):
        raise ValueError(
            f'n_samples must be a positive integer (Got {n_burnin})')
    if (thin <= 0) or (not isinstance(thin, int)):
        raise ValueError(f'thin must be a positive integer (Got {thin})')

    rs = check_random_state(random_state)
    w = 1.0  # initial bracket width

    n_samples_total = (n_samples + n_burnin) * thin

    samples = Parallel(n_jobs=n_jobs)(
        delayed(_slice_one_sample)(ptarget, x0, w, rs)
        for _ in range(n_samples_total))

    samples = np.array(samples)[(n_burnin * thin):][::thin]

    return samples

def _sample_parameter_r(n_samples, n_dim, sigma, random_state=None, n_jobs=1):
    """Sample the r parameters of a Riemannian Gaussian distribution.
    Sample the logarithm of the eigenvalues of a SPD matrix following a
    Riemannian Gaussian distribution.
    See https://arxiv.org/pdf/1507.01760.pdf for the mathematical details.
    Parameters
    ----------
    n_samples : int
        How many samples to generate.
    n_dim : int
        Dimensionality of the SPD matrices to be sampled.
    sigma : float
        Dispersion of the Riemannian Gaussian distribution.
    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible output across multiple function calls.
    n_jobs : int, default=1
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel. If -1 all CPUs are used.
    Returns
    -------
    r_samples : ndarray, shape (n_samples, n_dim)
        Samples of the r parameters of the Riemannian Gaussian distribution.
    """

    rs = check_random_state(random_state)
    x0 = rs.randn(n_dim)
    ptarget = partial(_pdf_r, sigma=sigma)
    r_samples = _slice_sampling(
        ptarget,
        n_samples=n_samples,
        x0=x0,
        random_state=random_state,
        n_jobs=n_jobs,
        )

    return r_samples



def _sample_parameter_U(n_samples, n_dim, random_state=None):
    """Sample the U parameters of a Riemannian Gaussian distribution.
    Sample the eigenvectors of a SPD matrix following a Riemannian Gaussian
    distribution.
    See https://arxiv.org/pdf/1507.01760.pdf for the mathematical details.
    Parameters
    ----------
    n_samples : int
        How many samples to generate.
    n_dim : int
        Dimensionality of the SPD matrices to be sampled.
    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible output across multiple function calls.
    Returns
    -------
    u_samples : ndarray, shape (n_samples, n_dim)
        Samples of the U parameters of the Riemannian Gaussian distribution.
    """

    rs = check_random_state(random_state)
    u_samples = np.zeros((n_samples, n_dim, n_dim))
    for i in range(n_samples):
        A = rs.randn(n_dim, n_dim)
        Q, _ = np.linalg.qr(A)
        u_samples[i] = Q

    return u_samples


if __name__ == "__main__":
    print(_sample_parameter_U(2,3))

def _slice_one_sample(ptarget, x0, w, rs):
    """Slice sampling for one sample
    Parameters
    ----------
    ptarget : function with one input
        The target pdf to sample from or a multiple of it.
    x0 : array
        Initial state for the MCMC procedure. Note that the shape of this array
        defines the dimensionality n_dim of the data points to be sampled.
    w : float
        Initial bracket width.
    rs : int, RandomState instance or None
        Pass an int for reproducible output across multiple function calls.
    Returns
    -------
    sample : ndarray, shape (n_dim,)
        Sample from the target pdf.
    """
    xt = np.copy(x0)
    n_dim = len(x0)

    for i in range(n_dim):

        ei = np.zeros(n_dim)
        ei[i] = 1

        # step 1 : evaluate ptarget(xt)
        Px = ptarget(xt)

        # step 2 : draw vertical coordinate uprime ~ U(0, ptarget(xt))
        uprime_i = Px * rs.rand()

        # step 3 : create a horizontal interval (xl_i, xr_i) enclosing xt_i
        r = rs.rand()
        xl_i = xt[i] - r * w
        xr_i = xt[i] + (1-r) * w
        while ptarget(xt + (xl_i - xt[i]) * ei) > uprime_i:
            xl_i = xl_i - w
        while ptarget(xt + (xr_i - xt[i]) * ei) > uprime_i:
            xr_i = xr_i + w

        # step 4 : loop
        while True:
            xprime_i = xl_i + (xr_i - xl_i) * rs.rand()
            Px = ptarget(xt + (xprime_i - xt[i]) * ei)
            if Px > uprime_i:
                break
            else:
                if xprime_i > xt[i]:
                    xr_i = xprime_i
                else:
                    xl_i = xprime_i

        # store coordinate i of new sample
        xt = np.copy(xt)
        xt[i] = xprime_i

    return xt