import math

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import curve_fit


def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def rSquare(estimations, measureds):
    """ Compute the coefficient of determination of random data.
    This metric gives the level of confidence about the model used to model data"""
    SEE = ((np.array(measureds) - np.array(estimations)) ** 2).sum()
    mMean = np.array(measureds).sum() / float(len(measureds))
    dErr = ((np.array(measureds) - mMean) ** 2).sum()

    print(SEE)
    print(dErr)
    return 1 - (SEE / dErr)


x = [0.001, 0.199, 0.394, 0.556, 0.797, 0.891, 1.171, 1.128, 1.437, 1.525, 1.72, 1.703, 1.895, 2.003, 2.108, 2.408,
        2.424, 2.537, 2.647, 2.74, 2.957, 2.58, 3.156, 3.051, 3.043, 3.353, 3.4, 3.606, 3.659, 3.671, 3.75, 3.827,
        3.902, 3.976, 4.048, 4.018, 4.286, 4.353, 4.418, 4.382, 4.444, 4.485, 4.465, 4.6, 4.681, 4.737, 4.792, 4.845,
        4.909, 4.919, 5.1, ]

y = range(1, len(x)+1)

#x = np.array([10, 19, 30, 35, 51])
#y = np.array([1, 7, 20, 50, 79])


def f(a, b, x):
    return a * math.exp(b * x)


plt.plot(x, y,  '-', label='data')
#z = np.polyfit(x, np.log(y), 1,  w=np.sqrt(y))
z = scipy.optimize.curve_fit(lambda t, a, b: a*np.exp(b*t),  x,  y)
print(z)


y1 = []
print(z[0][1], z[0][0])

for xx in x:
    y1.append(f(z[0][0], z[0][1], xx))

print('rSquare', rSquare(y1, y))

for i in range(len(y)):
    print(y[i], y1[i])


plt.plot(x, y1,  '-', label='data')


plt.show()
