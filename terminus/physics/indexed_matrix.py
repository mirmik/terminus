
import numpy


class IndexedMatrix:
    def __init__(self, matrix, lidxs=None, ridxs=None):
        self.lidxs = lidxs
        self.ridxs = ridxs
        #self.matrix = scipy.sparse.lil_matrix(matrix)
        self.matrix = matrix
        if self.lidxs:
            self.index_of_lidxs = {idx: lidxs.index(idx) for idx in lidxs}
        if self.ridxs:
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

    def inv(self):
        return IndexedMatrix(scipy.sparse.linalg.inv(self.matrix), self.ridxs, self.lidxs)

    def solve(self, b):
        return numpy.linalg.solve(self.matrix, b.matrix)

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

    def unsparse(self):
        return self.matrix.toarray()

    def transpose(self):
        return IndexedMatrix(self.matrix.T, self.ridxs, self.lidxs)

    def accumulate_from(self, other):
        lidxs = [self.index_of_lidxs[i] for i in other.lidxs]
        ridxs = [self.index_of_ridxs[i] for i in other.ridxs]
        for i in range(len(lidxs)):
            for j in range(len(ridxs)):
                self.matrix[lidxs[i], ridxs[j]] += other.matrix[i, j]

    def __str__(self):
        return "Matrix:\r\n{}\r\nLeft Indexes: {}\r\nRight Indexes: {}\r\n".format(self.matrix, self.lidxs, self.ridxs)


class IndexedVector:
    def __init__(self, matrix, idxs):
        if isinstance(matrix, numpy.ndarray) and len(matrix.shape) == 1:
            matrix = matrix.reshape(matrix.shape[0], 1)
        self.matrix = matrix
        self.idxs = idxs
        #self.matrix = scipy.sparse.lil_matrix(matrix)
        self.index_of_idxs = {idx: idxs.index(idx) for idx in idxs}

    def __str__(self):
        return "Vector:\r\n{}\r\nIndexes: {}\r\n".format(self.matrix, self.idxs)

    def accumulate_from(self, other):
        idxs = [self.index_of_idxs[i] for i in other.idxs]
        for i in range(len(idxs)):
            self.matrix[idxs[i]] += other.matrix[i]