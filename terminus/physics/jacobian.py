#!/usr/bin/env python3

import numpy


class ScrewCommutator:
    def __init__(self, dim):
        self._dim = dim
        self._sources = [ScrewSource(self, i) for i in range(dim)]
        self._values = [0] * dim

    def sources(self):
        return self._sources

    def set_value(self, idx, value):
        self.values[idx] = value


class ScrewSource:
    def __init__(self, commutator, index):
        self.commutator = commutator
        self.index = index

    def screw(self):
        return self.commutator.screw_for(self.index)

    def set_value(self, value):
        self.commutator.set_value(self.index, value)


class Body:
    def __init__(self, position, mass, inertia):
        self.position = position
        self.mass = mass
        self.inertia = inertia
        self.commutator = ScrewCommutator()

    def screw_commutator(self):
        return self.commutator

    def jacobian(self):
        return IndexedMatrix(self.diag([1, 1, 1, 1, 1, 1]), lidxs=None, ridxs=[id()])

    def mass_matrix(self):
        return create_matrix_of_mass(mass=self.mass, inertia=self.inertia)
