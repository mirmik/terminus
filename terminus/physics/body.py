#!/usr/bin/env python3

import numpy
from terminus.physics.screw_commutator import ScrewCommutator
from terminus.physics.indexed_matrix import IndexedMatrix, IndexedVector
from terminus.ga201 import Motor2
from terminus.ga201 import Screw2


class Body:
    def __init__(self, dim) -> None:
        self.dim = dim

    def left_velocity(self):
        return self._left_velocity

    def right_velocity(self):
        return self._left_velocity.inverted_kinematic_carry(self._position)

    def position(self):
        return self._position

    def right_mass_matrix(self):
        return self._mass_matrix

    def indexed_right_mass_matrix(self):
        return IndexedMatrix(self.right_mass_matrix(), None, None)

    def left_mass_matrix(self):
        raise NotImplemented

    def set_position(self, pos):
        self._position = pos

    def screw_commutator(self):
        return self.commutator

    def indexed_jacobian(self):
        return IndexedMatrix(numpy.diag([1, 1, 1]), lidxs=None, ridxs=self._commutator.sources())

    def jacobian(self):
        return numpy.diag([1, 1, 1])

    def create_matrix_of_mass(self, mass, inertia):
        A = inertia
        B = numpy.zeros((2, 1))
        C = numpy.zeros((1, 2))
        D = numpy.diag((mass,))

        return numpy.block([
            [A, B],
            [C, D]
        ])

    # def apply_speed(self, delta):
    #    increment = self._velocity * delta
    #    print(increment)
    #    self.set_position(self._position * Motor2.from_screw(increment))

    def set_left_velocity(self, vel):
        self._left_velocity = vel

    def set_right_velocity(self, vel):
        self._left_velocity = vel.kinematic_carry(self.position())

    def right_kinetic_screw(self):
        return Screw2.from_array(self._mass_matrix @ self.right_velocity().toarray())

    def left_kinetic_screw(self):
        rscrew = self.right_kinetic_screw()
        return rscrew.force_carry(self.position())

    def computation_indexes(self):
        return self._commutator.sources()


class Body2(Body):
    def __init__(self, position=Motor2(), velocity=Screw2(), mass=1, inertia=numpy.diag([1, 1])):
        super().__init__(dim=2)
        self._position = position
        self._left_velocity = velocity
        self._mass_matrix = self.create_matrix_of_mass(mass, inertia)
        self._commutator = ScrewCommutator(3)


if __name__ == "__main__":
    body = Body2(mass=4)
    M = body.right_mass_matrix()
    J = body.jacobian()
    print(J)
    print(body.indexed_jacobian())

    body.set_left_velocity(Screw2(v=numpy.array([1, 1]), m=1))

    print(M)
    print(body.position())
    print(body.right_kinetic_screw())
    print(body.left_kinetic_screw())

    print(body.left_velocity())
    print(body.right_velocity())

    A = body.indexed_right_mass_matrix()
    J = body.indexed_jacobian()
    JAJ = J.transpose() @ A @ J
    C = IndexedVector(numpy.array([0, 0, 1]), idxs=body.computation_indexes())

    from terminus.solver import quadratic_problem_solver_indexes_array
    x, l = quadratic_problem_solver_indexes_array([JAJ], [C])

    print(x, l)
