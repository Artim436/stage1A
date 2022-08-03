#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# X = np.array([i/100 for i in range(200)])

# Y = np.exp(X)-np.exp(-X)

# Z = np.exp(X)

# plt.plot(X, Y, label="target")
# plt.plot(X, Z, label="exp")
# plt.legend()
# plt.show()


a1, a2 = multivariate_normal.rvs(np.array([0, 0]), np.array([[1, 0], [0, 1]]), size=10).T

print(a1)
print(a1[0])