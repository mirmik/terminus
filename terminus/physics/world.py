

from terminus.solver import quadratic_problem_solver_indexes_array
from terminus.ga201 import Screw2

class World:
    def __init__(self):
        self.bodies = []
        self._iteration_counter = 0
        self._gravity = Screw2(v=[0,-1])

    def gravity(self):
        return self._gravity

    def add_body(self, body):
        self.bodies.append(body)
        body.bind_world(self)

    def integrate(self, delta):
        for body in self.bodies:
            body.integrate(delta)

    def main_right_parts(self):
        arr = []
        for body in self.bodies:
            arr.extend(body.forces_in_right_part())
        return arr

    def iteration(self, delta):
        A_list = [ body.main_matrix() for body in self.bodies ]
        C_list = self.main_right_parts()

        x,l = quadratic_problem_solver_indexes_array(A_list, C_list)
        x.upbind_values()
        
        for body in self.bodies:
            body.downbind_left_acceleration()
            body.integrate(0.01)

        self._iteration_counter += 1

    def iteration_counter(self):
        return self._iteration_counter