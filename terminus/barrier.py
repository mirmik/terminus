#/usr/bin/env python3

import math

def shotki_barrier(b, l):
    def func(x):
        if x >= l: 
            return 0
        return b/x + b*x/(l**2) - 2*b/l
    return func