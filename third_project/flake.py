#!/usr/bin/env python3

import numpy as np
from sklearn.utils import check_random_state
from scipy.stats import multivariate_normal
from joblib import Parallel, delayed


def rejection_sampling_a(sigma, r_sample):
    """ side function used for the rejection sampling
    algorithm in the case where we generate r with
        the first multivariate normal pdf
    Parameters
    ----------
    sigma : float
        Dispersion of the Riemannian Gaussian distribution.
    r_samples : ndarray, shape (1, n_dim)
        Sample of the r parameters of the Riemannian Gaussian distribution.
    Returns
    -------
    probability_of_acceptation : float
    """
    MU_A = np.array([sigma**2/2, -sigma**2/2])
    COV_MATRIX = (sigma**2)*np.eye(2)
    M = np.pi*(sigma**2)*np.exp(sigma**2/4)
    if r_sample[0] >= r_sample[1]:
        M = np.pi*(sigma**2)*np.exp(sigma**2/4)
        num = np.exp(-1/(2*sigma**2) * np.sum(r_sample**2))\
            * np.sinh((r_sample[0] - r_sample[1])/2) * 1/M
        den = multivariate_normal.pdf(r_sample, mean=MU_A, cov=COV_MATRIX)
        return num / den
    return 0


def rejection_sampling_b(sigma, r_sample):
    """ side function used for the rejection sampling
    algorithm in the case where we generate r with
        the second multivariate normal pdf
    Parameters
    ----------
    sigma : float
        Dispersion of the Riemannian Gaussian distribution.
    r_samples : ndarray, shape (n_samples, n_dim)
        Samples of the r parameters of the Riemannian Gaussian distribution.
    Returns
    -------
    probability_of_acceptation : float
    """
    MU_B = np.array([-sigma**2/2, sigma**2/2])
    COV_MATRIX = (sigma**2)*np.eye(2)
    M = np.pi*(sigma**2)*np.exp(sigma**2/4)
    if r_sample[0] < r_sample[1]:
        M = np.pi*(sigma**2)*np.exp(sigma**2/4)
        num = np.exp(-1/(2*sigma**2) * np.sum(r_sample**2))\
            * np.sinh((r_sample[1] - r_sample[0])/2)
        den = multivariate_normal.pdf(r_sample, mean=MU_B, cov=COV_MATRIX)*M
        return num/den
    return 0


def rejection_sampling_v1(n_samples, sigma):
    """ rejection sampling algorithm optimized for spatial
    complexity but not for time complexity works very well
    for low sigma values
    Parameters
    ----------
    n_samples : int
        Number of samples to get from the ptarget distribution.
    sigma : float
        Dispersion of the Riemannian Gaussian distribution.
    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible output across multiple function calls.
    Returns
    -------
    r_samples : ndarray, shape (n_samples, n_dim)
        Samples of the r parameters of the Riemannian Gaussian distribution.
    """
    MU_A = np.array([sigma**2/2, -sigma**2/2])
    MU_B = np.array([-sigma**2/2, sigma**2/2])
    COV_MATRIX = (sigma**2)*np.eye(2)
    RES = []
    cpt = 0
    while cpt != n_samples:
        if np.random.binomial(1, 0.5, 1) == 1:
            r_sample = multivariate_normal.rvs(MU_A, COV_MATRIX, 1)
            res = rejection_sampling_a(sigma, r_sample)
            if np.random.rand(1) < res:
                RES.append(r_sample)
                cpt += 1
        else:
            r_sample = multivariate_normal.rvs(MU_B, COV_MATRIX, 1)
            res = rejection_sampling_b(sigma, r_sample)
            if np.random.rand(1) < res:
                RES.append(r_sample)
                cpt += 1
    return np.array(RES)


def rejection_sampling_v2(n_samples, sigma, random_state=None):
    """ rejection sampling algorithm
    optimized for time complexity but not for spatial complexity,
    works very well for any sigma values which is not too low (<0.01)
    Parameters
    ----------
    n_samples : int
        Number of samples to get from the ptarget distribution.
    sigma : float
        Dispersion of the Riemannian Gaussian distribution.
    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible output across multiple function calls.
    Returns
    -------
    r_samples : ndarray, shape (n_samples, n_dim)
        Samples of the r parameters of the Riemannian Gaussian distribution.
    """
    N = int(n_samples/rejection_acceptation_probability_shifted(sigma) + 1000)
    MU_A = np.array([sigma**2/2, -sigma**2/2])
    MU_B = np.array([-sigma**2/2, sigma**2/2])
    COV_MATRIX = (sigma**2)*np.eye(2)
    RES = []
    cpt = 0
    rs = np.random.RandomState(random_state)
    B = rs.binomial(1, 0.5, N)
    M1 = multivariate_normal.rvs(MU_A, COV_MATRIX, N,
                                 random_state=random_state)
    M2 = multivariate_normal.rvs(MU_B, COV_MATRIX, N,
                                 random_state=random_state)
    R = rs.rand(N)
    icount_br = 0
    kcount_m1 = 0
    lcount_m2 = 0
    while cpt != n_samples:
        if B[icount_br] == 1:
            r_sample = M1[kcount_m1]
            kcount_m1 += 1
            res = rejection_sampling_a(sigma, r_sample)
            if R[icount_br] < res:
                RES.append(r_sample)
                cpt += 1
            icount_br += 1
            if (icount_br >= N) and (cpt != n_samples):
                raise ValueError("sigma value too low")
        else:
            r_sample = M2[lcount_m2]
            lcount_m2 += 1
            res = rejection_sampling_b(sigma, r_sample)
            if R[icount_br] < res:
                RES.append(r_sample)
                cpt += 1
            icount_br += 1
            if (icount_br >= N) and (cpt != n_samples):
                raise ValueError("sigma value too low")
    return np.array(RES)


def rejection_acceptation_probability(t):
    """polynomial function which approaches the probability
    of acceptation of the rejection_sampling algorithm depending on sigma
    Parameters
    ----------
    t : float
    Returns
    -------
    rejection_acceptation_probability(t) : float
    """
    return 0.01265*t**3 - 0.1648*t**2 + 0.7145*t-0.03374


def rejection_acceptation_probability_shifted(t):
    """polynomial function which approach the probability
    of acceptation of the rejection_sampling algorithm depending on sigma,
    the shift is used to estimate with confidence the number
    of samples we need to get n_samples
    Parameters
    ----------
    t : float
    Returns
    -------
    rejection_acceptation_probability_shifted(t) : float
    """
    if t <= 0.3:
        return 0.02
    return 0.01265*t**3 - 0.1648*t**2 + 0.7145*t-0.15


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


def rejection_sampling(n_samples, sigma, random_state=None):
    """ final rejection sampling algorithm
    Parameters
    ----------
    n_samples : int
        Number of samples to get from the ptarget distribution.
    sigma : float
        Dispersion of the Riemannian Gaussian distribution.
    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible output across multiple function calls.
    Returns
    -------
    r_samples : ndarray, shape (n_samples, n_dim)
        Samples of the r parameters of the Riemannian Gaussian distribution.
    """
    if sigma < 0.1:
        return rejection_sampling_v1(n_samples, sigma)
    else:
        return rejection_sampling_v2(n_samples, sigma,
                                     random_state=random_state)
