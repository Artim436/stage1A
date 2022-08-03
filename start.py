#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import random as rd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils import check_random_state
from scipy.stats import multivariate_normal


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
print(np.shape(R))

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



# #!/usr/bin/env python3

# import numpy as np
# import matplotlib.pyplot as plt
# import random as rd
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.utils import check_random_state
# from scipy.stats import multivariate_normal


# def pdf_r(r, sigma):
#     """Pdf for the log of eigenvalues of a SPD matrix.
#         Probability deπnsity function for the logarithm of the eigenvalues of a SPD
#         matrix samples from the Riemannian Gaussian distribution. See Said et al.
#         "Riemannian Gaussian distributions on the space of symmetric positive
#         definite matrices" (2017) for the mathematical details.
#         Parameters
#         ----------
#         r : ndarray, shape (n_dim,)
#             Vector with the logarithm of the eigenvalues of a SPD matrix.
#         sigma : float
#             Dispersion of the Riemannian Gaussian distribution.
#         Returns
#         -------
#         p : float
#             Probability density function applied to data point r.
#         """
#     if (sigma <= 0):
#         raise ValueError(f'sigma must be a positive number (Got {sigma})')

#     n_dim = len(r)
#     partial_1 = -np.sum(r**2) / (2*sigma**2)
#     partial_2 = 0
#     for i in range(n_dim):
#         for j in range(i + 1, n_dim):
#             partial_2 = partial_2 + np.log(np.sinh(np.abs(r[i] - r[j]) / 2))

#     return np.exp(partial_1 + partial_2)

# def _sample_parameter_U(n_samples, n_dim, random_state=None):
#     """Sample the U parameters of a Riemannian Gaussian distribution.
#     Sample the eigenvectors of a SPD matrix following a Riemannian Gaussian
#     distribution.
#     See https://arxiv.org/pdf/1507.01760.pdf for the mathematical details.
#     Parameters
#     ----------
#     n_samples : int
#         How many samples to generate.
#     n_dim : int
#         Dimensionality of the SPD matrices to be sampled.
#     random_state : int, RandomState instance or None, default=None
#         Pass an int for reproducible output across multiple function calls.
#     Returns
#     -------
#     u_samples : ndarray, shape (n_samples, n_dim)
#         Samples of the U parameters of the Riemannian Gaussian distribution.
#     """

#     u_samples = np.zeros((n_samples, n_dim, n_dim))
#     rs = check_random_state(random_state)
#     for i in range(n_samples):
#         A = rs.randn(n_dim, n_dim)
#         Q, _ = np.linalg.qr(A)
#         u_samples[i] = Q

#     return u_samples


# def pdf_joint_norm_dis(r, mu, sigma):
#     D = len(r)
#     S = np.eye(D) * (7*sigma)
#     x = 1/(2 * np.pi)**(D/2)
#     x *= 1/np.linalg.det(S)**(1/2)
#     x *= np.exp(-1/2 * np.dot(np.transpose(r), np.dot(np.linalg.inv(S), r)))
#     return x


# def rejection_sampling_A(f, g, sigma):
#     M = np.pi*np.exp((sigma**2)/4)*(sigma**2)
#     mu = np.array([(sigma**2)/2, -(sigma**2)/2])
#     cov_matrix = np.array([[sigma**2, 0], [0, sigma**2]])
#     r1, r2 = g.rvs(mu, cov_matrix, 1).T
#     if r1 >= r2:
#         r = np.array([r1, r2])
#         if rd.random() < f(r, sigma) / M * g.pdf(r, mu, cov_matrix):
#             return r

# def rejection_sampling_B(f, g, sigma):
#     M = np.pi*np.exp((sigma**2)/4)*(sigma**2)
#     mu = np.array([-(sigma**2)/2, (sigma**2)/2])
#     cov_matrix = np.array([[sigma**2, 0], [0, sigma**2]])
#     r1, r2 = g.rvs(mu, cov_matrix, 1).T
#     if r1 < r2:
#         r = np.array([r1, r2])
#         if rd.random() < f(r, sigma) / M * g.pdf(r, mu, cov_matrix):
#             return r


# def rejection_sampling(f, g, sigma, n):
#     R = []
#     while len(R) != n:
#         if rd.random()<0.5:
#             r = rejection_sampling_A(f, g, sigma)
#             if r is not None:
#                 R.append(r)
#         else:
#             r = rejection_sampling_B(f, g, sigma)
#             if r is not None:
#                 R.append(r)
#     return np.array(R)


# R = rejection_sampling(pdf_r, multivariate_normal, 2, 10)
# print(R)

# sigma = 1

# T = 5
# rx = np.linspace(-T, T, 200)
# ry = np.linspace(-T, T, 200)
# RX, RY = np.meshgrid(rx, ry)

# F = np.exp(-1/(2*sigma**2)*(RX**2 + RY**2)) * np.sinh(np.abs(RX - RY)/2)

# fig, ax = plt.subplots(figsize=(12, 6), ncols=2)
# ax[0].contour(RX, RY, F)
# ax[0].scatter(R[:,0],R[:,1], s=2, alpha=0.5)
# ax[1].hexbin(R[:,0], R[:,1], extent=(-T, T, -T, T))
# plt.show()

# # R1 = np.array([R[i][0] for i in range(len(R))])
# # R2 = np.array([R[i][1] for i in range(len(R))])
# # r1 = np.exp(R1)
# # r2 = np.exp(R2)

# # F = []
# # for k in range(len(R1)):
# #     O = _sample_parameter_U(1, 2)[0]
# #     F.append(np.linalg.multi_dot([O.T,np.diag(np.array([r1[k], r2[k]])),O]))

# # f1 = np.array([F[i][0][0] for i in range(len(F))])
# # f2 = np.array([F[i][1][1] for i in range(len(F))])
# # f3 = np.array([F[i][0][1] for i in range(len(F))])

# # fig = plt.figure(figsize=(10,10))

# # ax = Axes3D(fig)
# # ax.scatter3D(f1, f2, f3)
# # ax.set_xlim(0, 10)
# # ax.set_ylim(0, 10)
# # ax.set_zlim(-3, 3)
# # ax.set_xlabel('x')
# # ax.set_ylabel('y')
# # ax.set_zlabel('z')
# # plt.show()