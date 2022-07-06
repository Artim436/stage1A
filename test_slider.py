from mpl_toolkits.mplot3d import axes3d

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

import numpy as  np
from pyriemann.datasets import sample_gaussian_spd, generate_random_spd_matrix
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np
from scipy.linalg import sqrtm
from numpy.linalg import inv
from numpy.linalg import matrix_power
from numpy.linalg import multi_dot
from numpy.linalg import eig



n_matrices = 30 # how many SPD matrices to generate
n_dim = 2 # number of dimensions of the SPD matrices
sigma = 1.0  # dispersion of the Gaussian distribution
epsilon = 4.0  # parameter for controlling the distance between centers
random_state = 42  # ensure reproducibility

mean = np.eye(2)


dict = {}
dict["sample_1"] = sample_gaussian_spd(n_matrices, mean, 1.0, random_state)
dict["sample_2"] = sample_gaussian_spd(n_matrices, mean, 1.25, random_state)
dict["sample_3"] = sample_gaussian_spd(n_matrices, mean, 1.50, random_state)
dict["sample_4"] = sample_gaussian_spd(n_matrices, mean, 1.75, random_state)

# x = [sample_1[i][0][0] for i in range(n_matrices)]
# y = [sample_1[i][1][1] for i in range(n_matrices)]
# z = [sample_1[i][0][1] for i in range(n_matrices)]



# The parametrized function to be plotted

def f(i, sigma):
    S = [1.0, 1.25, 1.50, 1.75]
    s = S.index(sigma) + 1
    sample  = dict.get(f"sample_{s}")
    x = []
    y = []
    z = []
    for elt in i:
        x.append(sample[elt][0][0])
        y.append(sample[elt][1][1])
        z.append(sample[elt][0][1])
    X = np.asarray(x)
    Y = np.asarray(y)
    Z = np.asarray(z)
    return X, Y, Z

i = np.linspace(1, n_matrices, n_matrices)

# Define initial parameters
init_sigma = 1.0

# Create the figure and the line that we will manipulate
ax = plt.axes(projection='3d')
ax.scatter3D(f(i, init_sigma)[0], f(i, init_sigma)[1], f(i, init_sigma)[2])

# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axsigma = plt.axes([0.25, 0.1, 0.65, 0.03])
sigma_slider = Slider(
    ax=axsigma,
    label='Sigma',
    valmin=1.0,
    valmax=2.0,
    valinit=init_sigma,
)

# The function to be called anytime a slider's value changes
def update(val):
    line.set_ydata(f(i, sigma_slider.val))
    fig.canvas.draw_idle()


# register the update function with each slider
sigma_slider.on_changed(update)

# # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
# resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
# button = Button(resetax, 'Reset', hovercolor='0.975')


# def reset(event):
#     sigma_slider.reset()
# button.on_clicked(reset)


plt.show()