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


def full_indexes_list_matrix(arr):
    l = set()
    r = set()
    for a in arr:
        for index in a.lidxs:
            l.add(index)
        for index in a.ridxs:
            r.add(index)
    return sorted(list(l)), sorted(list(r))


def indexed_matrix_summation(arr):
    lidxs, ridxs = full_indexes_list_matrix(arr)
    result_matrix = IndexedMatrix(numpy.zeros(
        (len(lidxs), len(ridxs))), lidxs, ridxs)
    for m in arr:
        result_matrix.accumulate_from(m)
        print(result_matrix)
    return result_matrix


def indexed_vector_summation(arr):
    print(arr)
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
    print(*Aarr)
    print(*Carr)
    A = indexed_matrix_summation(Aarr)
    B = indexed_matrix_summation(Barr)
    C = indexed_vector_summation(Carr)
    D = indexed_vector_summation(Darr)
    x, l = quadratic_problem_solver_indexes(A=A, B=B, C=C, D=D)
    return x, l


def quadratic_problem_solver_indexes(A: IndexedMatrix, C: IndexedVector, B: IndexedMatrix, D: IndexedVector):
    print(A.matrix)
    print(C.matrix)
    x = numpy.linalg.pinv(A.matrix) @ C.matrix
    l = 0
    return IndexedVector(x, idxs=A.ridxs), IndexedVector(l, idxs=B.ridxs)


def quadratic_problem_solver(A, B, C, D):
    """
        [A]x + [B]l = C
        [B^t]x = D
    """
    return x, l


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
