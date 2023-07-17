#!/usr/bin/env python3

import numpy
import scipy


class IndexedMatrix:
    def __init__(self, matrix, lidxs=None, ridxs=None):
        self.lidxs = lidxs
        self.ridxs = ridxs
        self.matrix = scipy.sparse.lil_matrix(matrix)
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

    def raise_if_lidxs_is_not_same(self, oth):
        if self.lidxs != oth.lidxs:
            raise Exception("indexes is not same in convolution")

    def raise_if_ridxs_is_not_same(self, oth):
        if self.ridxs != oth.ridxs:
            raise Exception("indexes is not same in convolution")

    def raise_if_not_linkaged(self, oth):
        if self.ridxs != oth.lidxs:
            raise Exception("indexes is not same in convolution")

    def __add__(self, oth):
        self.raise_if_lidxs_is_not_same(oth)
        self.raise_if_ridxs_is_not_same(oth)
        return IndexedMatrix(self.matrix + oth.matrix, self.lidxs, self.ridxs)

    def transpose(self):
        return IndexedMatrix(self.matrix.T, self.ridxs, self.lidxs)

    def accumulate_from(self, other):
        lidxs = [self.index_of_lidxs[i] for i in other.lidxs]
        ridxs = [self.index_of_ridxs[i] for i in other.ridxs]
        self.matrix[lidxs, ridxs]

    def __str__(self):
        return "Matrix:\r\n{}\r\nLeft Indexes: {}\r\nRight Indexes: {}\r\n".format(self.matrix, self.lidxs, self.ridxs)


class IndexedVector:
    def __init__(self, matrix, idxs):
        if isinstance(matrix, numpy.ndarray) and len(matrix.shape) == 1:
            matrix = matrix.reshape(matrix.shape[0], 1)
        self.idxs = idxs
        self.matrix = scipy.sparse.lil_matrix(matrix)
        self.index_of_idxs = {idx: idxs.index(idx) for idx in idxs}

    def __str__(self):
        return "Vector:\r\n{}\r\nIndexes: {}\r\n".format(self.matrix, self.idxs)

    def accumulate_from(self, other):
        idxs = [self.index_of_idxs[i] for i in other.idxs]
        self.matrix[idxs] += other.matrix


def create_matrix_of_mass(self, mass, inertia):
    indexed_matrix = IndexedMatrix(numpy.block(
        [[inertia, numpy.zeros((3, 3))],
         [numpy.zeros((3, 3)), numpy.diag(mass, mass, mass)]]
    ), None, None)
    return indexed_matrix


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
    return result_matrix


def indexed_vector_summation(arr):
    idxs = full_indexes_list_vector(arr)
    result_matrix = IndexedVector(numpy.zeros(
        (len(idxs))), idxs)
    for m in arr:
        result_matrix.accumulate_from(m)
    return result_matrix


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
    B = IndexedMatrix(numpy.array([[1, 2, 0], [0, 1, 0], [0, 0, 1]]), [
        "a", "b", "c"], ["x", "y", "z"])

    C = indexed_matrix_summation([A, B])

    V1 = IndexedVector(numpy.array([0, 1, 4]), ["x", "y", "z"])
    V2 = IndexedVector(numpy.array([1, 1, 3]), ["a", "y", "z"])

    print(A + B)
    print(A @ V1)
    print(indexed_vector_summation([V1, V2]))
