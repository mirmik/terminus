import numpy

def linear_solve(A, b):
    im = numpy.linalg.pinv(A)
    res = im.dot(b)
    return res
