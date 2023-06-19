#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

def barrier(b, r):
    def func(x):
        return b/x + b*x/(r**2) - 2*b/r
    return func

def derivative_barrier(b,r):
    def func(x):
        return -b/(x**2) + b/(r**2)
    return func


bf = np.vectorize(barrier(0.1,5))
dbf = np.vectorize(derivative_barrier(0.1,5))

x = np.arange(4, 6, 0.1)
y = bf(x)
dy = dbf(x)

plt.plot(x,y, x,dy)
plt.grid()
plt.show()