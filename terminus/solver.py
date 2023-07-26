#!/usr/bin/env python3

import numpy
import scipy
from terminus.physics.indexed_matrix import IndexedMatrix, IndexedVector


def full_indexes_list_vector(arr):
    s = set()
    for a in arr:
        for index in a.idxs:
            s.add(index)
    return sorted(list(s))


def full_indexes_list_matrix(arr, lidxs=None, ridxs=None):
    l = set()
    r = set()
    for a in arr:
        if lidxs is None:
            for index in a.lidxs:
                l.add(index)
        if ridxs is None:
            for index in a.ridxs:
                r.add(index)
    
    lidxs = lidxs if lidxs is not None else sorted(list(l))
    ridxs = ridxs if ridxs is not None else sorted(list(r))

    return lidxs, ridxs


def indexed_matrix_summation(arr, lidxs=None, ridxs=None):
    lidxs, ridxs = full_indexes_list_matrix(arr, lidxs=lidxs, ridxs=ridxs)

    result_matrix = IndexedMatrix(numpy.zeros(
        (len(lidxs), len(ridxs))), lidxs, ridxs)
    for m in arr:
        result_matrix.accumulate_from(m)
    return result_matrix


def indexed_vector_summation(arr, idxs=None):
    if idxs is None:
        idxs = full_indexes_list_vector(arr)
    result_vector = IndexedVector(numpy.zeros(
        (len(idxs))), idxs)
    for m in arr:
        result_vector.accumulate_from(m)
    return result_vector


def invoke_set_values_for_indexed_vector(self, indexed_vector):
    indexes = indexed_vector.idxs
    values = indexed_vector.matrix
    for idx, val in zip(indexes, values):
        idx.set_value(val)

def quadratic_problem_solver_indexes_array(Aarr: list, Carr: list, Barr: list = [], Darr: list = []):
    A = indexed_matrix_summation(Aarr)

    if len(Carr) == 0:
        Carr = [IndexedVector([0] * len(A.lidxs), idxs=A.lidxs)]

    C = indexed_vector_summation(Carr, idxs=A.lidxs)

    if len(Barr) != 0:
        B = indexed_matrix_summation(Barr, lidxs=A.ridxs)
        D = indexed_vector_summation(Darr, idxs=B.ridxs)
    else:
        B = None
        D = None

    x, l = quadratic_problem_solver_indexes(A=A, B=B, C=C, D=D)
    return x, l


def quadratic_problem_solver_indexes(A: IndexedMatrix, C: IndexedVector, B: IndexedMatrix, D: IndexedVector):    
  
    C.matrix = C.matrix.reshape((C.matrix.shape[0], 1))
    if B is not None:
        D.matrix = D.matrix.reshape((D.matrix.shape[0], 1))  
        Q = numpy.block([[A.matrix, B.matrix], [B.matrix.T, numpy.zeros((len(B.ridxs), len(B.ridxs)))]])
        b = numpy.block([[C.matrix], [D.matrix]])
    else:
        Q = A.matrix
        b = C.matrix

    if A.lidxs != C.idxs:
        raise Exception("indexes is not same in convolution")

    if B.ridxs != D.idxs:
        raise Exception("indexes is not same in convolution")

    print("Equation:")
    print(Q)
    print(b)
    print(numpy.linalg.pinv(Q))

    X = numpy.linalg.pinv(Q) @ b
    X = X.reshape((X.shape[0],))
    x = X[:len(A.lidxs)]
    l = X[len(A.lidxs):]

    print("Answer:")
    print(x)
    print(l)
    
    if B is not None:
        return IndexedVector(x, idxs=A.ridxs), IndexedVector(l, idxs=B.ridxs)
    else:
        return IndexedVector(x, idxs=A.ridxs), None


if __name__ == "__main__":
    A = IndexedMatrix(numpy.array([[1, 2, 0], [0, 1, 0], [0, 0, 1]]), [
        "a", "b", "c"], ["x", "y", "z"])
    B = IndexedMatrix(numpy.array([[1, 2, 0], [0, 1, 0], [0, 0, 1]]), [
        "a", "b", "c"], ["x", "y", "z"])

    C = indexed_matrix_summation([A, B])

    V1 = IndexedVector(numpy.array([1, 1, 4]), ["x", "y", "z"])
    V2 = IndexedVector(numpy.array([1, 1, 3]), ["a", "y", "z"])

    print((((A).solve(V1))))
    print(A @ V1)
    print(indexed_vector_summation([V1, V2]))
