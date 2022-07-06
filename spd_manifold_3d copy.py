#!/usr/bin/env python3

import numpy as  np
from pyriemann.datasets import sample_gaussian_spd, generate_random_spd_matrix
import numpy as np
import plotly.graph_objects as go




n_matrices = 3000 # how many SPD matrices to generate
n_dim = 2 # number of dimensions of the SPD matrices
sigma = 1.3  # dispersion of the Gaussian distribution
epsilon = 4.0  # parameter for controlling the distance between centers
random_state = 42  # ensure reproducibility

mean = np.eye(2)

sample_1 = sample_gaussian_spd(n_matrices, mean, sigma, random_state)



x = [sample_1[i][0][0] for i in range(n_matrices)]
y = [sample_1[i][1][1] for i in range(n_matrices)]
z = [sample_1[i][0][1] for i in range(n_matrices)]

X = np.asarray(x)
Y = np.asarray(y)
Z = np.asarray(z)

fig = go.Figure(data=[go.Mesh3d(x=X, y=Y, z=Z, alphahull=5,
                   opacity=0.4)])

fig.show()