#!/usr/bin/env python3

import numpy
import scipy
from terminus.physics.indexed_matrix import IndexedMatrix, IndexedVector
import torch

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


def symmetric_full_indexes_list_matrix(arr, idxs=None):
    l = set()
    for a in arr:
        if idxs is None:
            for index in a.lidxs:
                l.add(index)

    idxs = idxs if idxs is not None else sorted(list(l))
    return idxs


def indexed_matrix_summation(arr, lidxs=None, ridxs=None):
    lidxs, ridxs = full_indexes_list_matrix(arr, lidxs=lidxs, ridxs=ridxs)

    result_matrix = IndexedMatrix(numpy.zeros(
        (len(lidxs), len(ridxs))), lidxs, ridxs)
    for m in arr:
        result_matrix.accumulate_from(m)
    return result_matrix


def symmetric_indexed_matrix_summation(arr, idxs=None):
    idxs = symmetric_full_indexes_list_matrix(
        arr, idxs=idxs)

    result_matrix = IndexedMatrix(numpy.zeros(
        (len(idxs), len(idxs))), idxs, idxs)
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


def qpc_solver_indexes_array(
        Aarr: list, 
        Carr: list, 
        Barr: list = [], 
        Darr: list = [],
        Harr: list = [],
        Ksiarr: list = []):
    A = symmetric_indexed_matrix_summation(Aarr)
    C = indexed_vector_summation(Carr, idxs=A.lidxs)
    B = indexed_matrix_summation(Barr, lidxs=A.ridxs)
    D = indexed_vector_summation(Darr, idxs=B.ridxs)
    H = indexed_matrix_summation(Harr, lidxs=A.ridxs)
    Ksi = indexed_vector_summation(Ksiarr, idxs=H.ridxs)

    x, l, ksi = qpc_solver_indexes(A=A, B=B, C=C, D=D, H=H, Ksi=Ksi)
    return x, l, ksi


def qpc_solver_indexes(
        A: IndexedMatrix, 
        C: IndexedVector, 
        B: IndexedMatrix, 
        D: IndexedVector,
        H: IndexedMatrix,
        Ksi: IndexedVector):
    C.matrix = C.matrix.reshape((C.matrix.shape[0], 1))
    D.matrix = D.matrix.reshape((D.matrix.shape[0], 1))
    Ksi.matrix = Ksi.matrix.reshape((Ksi.matrix.shape[0], 1))
    Z1 = numpy.zeros((len(B.ridxs), len(B.ridxs)))
    Z2 = numpy.zeros((len(H.ridxs), len(H.ridxs)))
    Z3 = numpy.zeros((len(B.ridxs), len(H.ridxs)))
    Q = numpy.block([
        [A.matrix, B.matrix, H.matrix], 
        [B.matrix.T, Z1, Z3],
        [H.matrix.T, Z3.T, Z2]
    ])
    b = numpy.block([[C.matrix], [D.matrix], [Ksi.matrix]])    

    Q_torch = torch.tensor(Q, dtype=torch.float64).cuda()
    b_torch = torch.tensor(b, dtype=torch.float64).cuda()

    #X = numpy.linalg.inv(Q) @ b
    X_torch = torch.linalg.solve(Q_torch, b_torch)
    #X = X_torch.cpu().detach().numpy()
    #X = numpy.linalg.solve(Q, b)

    X = X_torch.reshape((X_torch.shape[0],))
    x = X[:len(A.lidxs)]
    l = X[len(A.lidxs):len(A.lidxs) + len(B.ridxs)]
    ksi = X[len(A.lidxs) + len(B.ridxs):]
    
    x = x.cpu().detach().numpy()
    l = l.cpu().detach().numpy()
    ksi = ksi.cpu().detach().numpy()

    return IndexedVector(x, idxs=A.ridxs), IndexedVector(l, idxs=B.ridxs), IndexedVector(ksi, idxs=H.ridxs)


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
