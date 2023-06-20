#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

def barrier(b, l):
    def func(x):
        if x >= l: 
            return 0
        return b/x + b*x/(l**2) - 2*b/l
    return func

bf = np.vectorize(barrier(1,5))

x = np.arange(0.01, 10, 0.1)
y = bf(x)

plt.plot(x,y, 'k')
plt.ylim(-0.2,2)
plt.grid()
plt.xlabel("|r|")
plt.ylabel("B(|r|)")
plt.show()