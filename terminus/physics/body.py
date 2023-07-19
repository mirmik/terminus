#!/usr/bin/env python3

import numpy
from terminus.physics.screw_commutator import ScrewCommutator
from terminus.physics.indexed_matrix import IndexedMatrix, IndexedVector
from terminus.ga201 import Motor2
from terminus.ga201 import Screw2


class Body:
    def __init__(self, space_dim, dof) -> None:
        self.space_dim = space_dim
        self.dof = dof
        self._right_acceleration_global = Screw2()
        self._right_velocity_global = Screw2()
        self._resistance_coefficient = 0
        self._right_forces_global = []
        self._right_forces = []
        self._position = Motor2()
        self._world = None

    def bind_world(self, w):
        self._world = w

    def translation(self):
        return self._position.factorize_translation_vector()

    def rotation(self):
        return self._position.factorize_rotation_angle()

    def downbind_left_acceleration(self):
        self._left_acceleration = Screw2(
            m=self._variable_indexer.values()[0],
            v=numpy.array(self._variable_indexer.values()[1:])
        )

    def unbind_force(self, force):
        if force.is_right_global() and force.is_linked_to(self):
            self._right_forces_global.remove(force)
        elif force.is_right() and force.is_linked_to(self):
            self._right_forces.remove(force)
        else:
            raise Exception("Force is not linked to this body")

        force.clean_bind_information()

    def add_right_force_global(self, force):
        self._right_forces_global.append(force)
        force.set_right_global_type()
        force.set_linked_object(self)

    def add_right_force(self, force):
        self._right_forces.append(force)
        force.set_right_type()
        force.set_linked_object(self)

    def right_acceleration_global(self):
        return self._left_acceleration

    def right_velocity_global(self):
        return self._right_velocity_global

    def right_velocity(self):
        return self._right_velocity_global.inverse_rotate_by(self._position)

    def position(self):
        return self._position

    def right_mass_matrix(self):
        return self._mass_matrix

    def indexed_right_mass_matrix(self):
        return IndexedMatrix(self.right_mass_matrix(), None, None)

    def right_mass_matrix_global(self):
        motor_matrix = self._position.rotation_matrix()
        return motor_matrix.T @ self._mass_matrix @ motor_matrix

    def indexed_right_mass_matrix_global(self):
        return IndexedMatrix(self.right_mass_matrix(), 
            self.commutation_indexes(), 
            self.variable_indexes()
            )
    
    def main_matrix(self):
        return self.indexed_right_mass_matrix_global()

    def set_position(self, pos):
        self._position = pos

    def screw_commutator(self):
        return self.commutator

    def jacobian(self):
        return numpy.diag([1, 1, 1])

    def set_right_velocity_global(self, vel):
        self._right_velocity_global = vel

    def set_right_velocity(self, vel):
        self._right_velocity_global = vel.rotate_by(self.position())

    def right_kinetic_screw(self):
        return Screw2.from_array(self._mass_matrix @ self.right_velocity().toarray())

    def right_kinetic_screw_global(self):
        rscrew = self.right_kinetic_screw()
        return rscrew.rotate_by(self.position())

    def computation_indexes(self):
        return self._commutator.sources()

    def right_global_gravity(self):
        world_gravity = self._world.gravity()
        return IndexedVector(
            (world_gravity*self._mass).toarray(), 
            self.commutation_indexes())

    def right_global_resistance(self):
        return IndexedVector(
            - self._right_velocity_global.toarray() * self._resistance_coefficient,
            self.commutation_indexes()
        )

    def right_forces_global_as_indexed_vectors(self):
        arr = []
        for f in self._right_forces_global:
            arr.append(f.to_indexed_vector())
        return arr

    def right_forces_as_indexed_vectors(self):
        arr = []
        for f in self._right_forces:
            arr.append(f.to_indexed_vector_rotated_by(self._position))
        return arr

    def forces_in_right_part(self):
        return ([
            self.right_global_gravity(),
            self.right_global_resistance()
        ] 
           + self.right_forces_global_as_indexed_vectors() 
           + self.right_forces_as_indexed_vectors() 
        )

    def integrate(self, delta):
        self.set_right_velocity_global(
            self.right_velocity_global() + 
            self.right_acceleration_global() * delta
        )

        rvel = self.right_velocity() / 2
        self._position = self._position * Motor2.from_screw(rvel)
        self._position.self_normalize()

    def variable_indexes(self):
        return self._variable_indexer.sources()

    def commutation_indexes(self):
        return self._commutation_indexer.sources()


class Body2(Body):
    def __init__(self, mass=1, inertia=numpy.diag([1])):
        super().__init__(space_dim=2, dof=3)
        self._mass = mass
        self._mass_matrix = self.create_matrix_of_mass(mass, inertia)
        self._variable_indexer = ScrewCommutator(self.dof)
        self._commutation_indexer = ScrewCommutator(3)


    def create_matrix_of_mass(self, mass, inertia):
        A = inertia
        B = numpy.zeros((1, 2))
        C = numpy.zeros((2, 1))
        D = numpy.diag((mass,mass))

        return numpy.block([
            [A, B],
            [C, D]
        ])

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
