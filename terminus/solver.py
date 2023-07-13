import numpy


class IndexedMatrix:
    def __init__(self, matrix, lidxs=None, ridxs=None):
        self.lidxs = lidxs
        self.ridxs = ridxs
        self.matrix = matrix
        self.index_of_lidxs = {idx: lidxs.index(idx) for idx in lidxs}
        self.index_of_ridxs = {idx: ridxs.index(idx) for idx in ridxs}

    def __matmul__(self, oth):
        if self.iidxs or self.ridxs:
            if self.lidxs != self.ridxs:
                raise Exception("indexes is not same in convolution")
        matrix = self.matrix @ oth.matrix
        return IndexedMatrix(matrix, self.lidxs, oth.ridxs)

    def transpose(self):
        return IndexedMatrix(self.matrix.T, self.ridxs, self.lidxs)

    def accumulate_from(self, other):
        lidxs = [self.index_of_lidxs[i] for i in other.lidx]
        ridxs = [self.index_of_ridxs[i] for i in other.ridx]
        self.matrix[lidxs, ridxs]


class IndexedVector:
    def __init__(self, matrix, idxs):
        self.idxs = idxs
        self.matrix = matrix


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
