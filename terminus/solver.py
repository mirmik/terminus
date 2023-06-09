#!/usr/bin/env python3

import numpy


class IndexedMatrix:
    def __init__(self, matrix, lidxs=None, ridxs=None):
        self.lidxs = lidxs
        self.ridxs = ridxs
        self.matrix = matrix
        self.index_of_lidxs = {idx: lidxs.index(idx) for idx in lidxs}
        self.index_of_ridxs = {idx: ridxs.index(idx) for idx in ridxs}

    def matmul(self, oth):
        if self.ridxs != oth.lidxs:
            raise Exception("indexes is not same in convolution")
        matrix = self.matrix @ oth.matrix
        return IndexedMatrix(matrix, self.lidxs, oth.ridxs)

    def vecmul(self, oth):
        if self.ridxs != oth.idxs:
            raise Exception("indexes is not same in convolution")
        matrix = self.matrix @ oth.matrix
        return IndexedVector(matrix, self.lidxs)

    def __matmul__(self, oth):
        if isinstance(oth, IndexedVector):
            return self.vecmul(oth)
        else:
            return self.matmul(oth)

    def transpose(self):
        return IndexedMatrix(self.matrix.T, self.ridxs, self.lidxs)

    def accumulate_from(self, other):
        lidxs = [self.index_of_lidxs[i] for i in other.lidx]
        ridxs = [self.index_of_ridxs[i] for i in other.ridx]
        self.matrix[lidxs, ridxs]

    def __str__(self):
        return "{} {} {}".format(self.matrix, self.lidxs, self.ridxs)


class IndexedVector:
    def __init__(self, matrix, idxs):
        self.idxs = idxs
        self.matrix = matrix

    def __str__(self):
        return "{} {}".format(self.matrix, self.idxs)


def create_matrix_of_mass(self, mass, inertia):
    indexed_matrix = IndexedMatrix(numpy.block(
        [[inertia, numpy.zeros((3, 3))],
         [numpy.zeros((3, 3)), numpy.diag(mass, mass, mass)]]
    ), None, None)
    return indexed_matrix


def full_indexes_list(arr):
    s = set()
    for a in arr:
        for index in a.idxs:
            s.insert(index)
    return list(s)


def indexed_matrix_summation(arr):
    lidxs, ridxs = full_indexes_list(arr)
    result_matrix = IndexedMatrix(numpy.zeros(
        (len(lidxs), len(ridxs))), lidxs, ridxs)
    for m in arr:
        result_matrix.accumulate_from(m)
    return result_matrix


def indexed_vector_summation(arr):
    return sum(arr)


def invoke_set_values_for_indexed_vector(self, indexed_vector):
    indexes = indexed_vector.idxs
    values = indexed_vector.matrix
    for idx, val in zip(indexes, values):
        idx.set_value(val)


def quadratic_problem_solver_indexes_array(Aarr: list, Barr: list, Carr: list, Darr: list):
    A = indexed_matrix_summation(Aarr)
    B = indexed_matrix_summation(Barr)
    C = indexed_vector_summation(Carr)
    D = indexed_vector_summation(Darr)
    x, l = quadratic_problem_solver_indexes(A, B, C, D)
    return IndexedVector(x, indexes=A.ridxs), IndexedVector(l, indexes=B.ridxs)


def quadratic_problem_solver_indexes(A: IndexedMatrix, B: IndexedMatrix, C: IndexedVector, D: IndexedVector):
    pass


def quadratic_problem_solver(A, B, C, D):
    """
        [A]x + [B]l = C
        [B^t]x = D  
    """
    return x, l


if __name__ == "__main__":
    A = IndexedMatrix(numpy.array([[1, 2, 0], [0, 1, 0], [0, 0, 1]]), [
        "a", "b", "c"], ["x", "y", "z"])

    V = IndexedVector(numpy.array([0, 1, 0]), ["x", "y", "z"])

    print(A @ V)
