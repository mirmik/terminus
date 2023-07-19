

class ScrewCommutator:
    def __init__(self, dim):
        self._dim = dim
        self._sources = [ScrewSource(self, i) for i in range(dim)]
        self._values = [0] * dim

    def sources(self):
        return self._sources

    def set_value(self, idx, value):
        self._values[idx] = value

    def values(self):
        return self._values

class ScrewSource:
    def __init__(self, commutator, index):
        self.commutator = commutator
        self.index = index

    def screw(self):
        return self.commutator.screw_for(self.index)

    def set_value(self, value):
        self.commutator.set_value(self.index, value)

    def __str__(self):
        return str(id(self))

    def __repr__(self):
        return str(id(self))

    def __lt__(self, oth):
        return id(self) < id(oth)
