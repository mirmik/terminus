import numpy

def linear_solve(A, b):
    im = numpy.linalg.pinv(A)
    res = im.dot(b)
    return res


def nullspace(A):
    I = numpy.identity(A.shape[1])
    A_plus = numpy.linalg.pinv(A)
    N = I - A_plus.dot(A)
    return N
   