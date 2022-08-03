#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


l = 100000



Y = {"Y0": [], "Y1": [], "Y2": []}


N = [1, 2, 10]
for n in range(3):
    X = np.random.uniform(0,1, N[n])
    for k in range(l):
        for i in range(len(X)):
            Y[f"Y{n}"].append(sum([int(X[i]*100)/100 for i in range(len(X))])/len(X))
        X = np.random.uniform(0,1, N[n])
    





plt.subplot(311)
x = [i/100 for i in range(101)]
munorm, stdnorm = norm.fit(np.asarray(Y["Y0"]))
plt.hist(Y["Y0"], bins=x, normed=True)
plt.ylim(0,1.5)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
pnorm = norm.pdf(x, munorm, stdnorm)
plt.plot(x, pnorm, 'k', linewidth=2)
title = "Fit Values: munorm {:.2f} and stdnorm {:.2f}".format(munorm, stdnorm)
plt.title(title)

plt.subplot(312)
x = [i/100 for i in range(101)]
munorm, stdnorm = norm.fit(np.asarray(Y["Y1"]))
plt.hist(Y["Y1"], bins=x, normed=True)
plt.ylim(0,3)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
pnorm = norm.pdf(x, munorm, stdnorm)
plt.plot(x, pnorm, 'k', linewidth=2)
title = "Fit Values: munorm {:.2f} and stdnorm {:.2f}".format(munorm, stdnorm)
plt.title(title)

plt.subplot(313)
x = [i/100 for i in range(101)]
munorm, stdnorm = norm.fit(np.asarray(Y["Y2"]))
plt.hist(Y["Y2"], bins=x, normed=True)
plt.ylim(0, 5)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
pnorm = norm.pdf(x, munorm, stdnorm)
plt.plot(x, pnorm, 'k', linewidth=2)
title = "Fit Values: munorm {:.2f} and stdnorm {:.2f}".format(munorm, stdnorm)
plt.title(title)


plt.show()