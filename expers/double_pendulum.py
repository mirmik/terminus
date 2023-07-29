#!/usr/bin/env python3

import time
from terminus.physics.indexed_matrix import IndexedVector
from terminus.physics.body import Body2
from terminus.solver import quadratic_problem_solver_indexes_array
from terminus.physics.world import World
from terminus.physics.force import Force
from terminus.physics.force_link import VariableMultiForce
from terminus.ga201 import Screw2
from terminus.ga201 import Motor2
import numpy

import zencad

#numpy.set_printoptions(precision=1, suppress=True)

body1 = Body2(mass=1)
body2 = Body2()

world = World()
world.add_body(body1)
world.add_body(body2)
world.set_gravity(Screw2(v=[0, -10]))

body1.set_resistance_coefficient(0)
body2.set_resistance_coefficient(0)

body1.set_position(Motor2.translation(0, -10))
body2.set_position(Motor2.translation(10, -10))

force_link1 = VariableMultiForce(child=body1, parent=None, position=Motor2.translation(0, 0), senses=[
    Screw2(v=[1, 0]), Screw2(v=[0, 1])], stiffness=[1, 1], use_child_frame=False)
world.add_link_force(force_link1)

force_link2 = VariableMultiForce(child=body2, parent=body1, position=Motor2.translation(0, -10), senses=[
    Screw2(v=[1, 0]), Screw2(v=[0, 1])], stiffness=[1, 1], use_child_frame=False)
world.add_link_force(force_link2)

sph0 = zencad.disp(zencad.sphere(1))
sph1 = zencad.disp(zencad.sphere(1))

start_time = time.time()
planned_time = start_time

#while True:
def animate(wdg):
    global planned_time
    current_time = time.time()
    world.iteration(0.01)
    #world.correction()

    sph0.relocate(zencad.translate(body1.translation()[0], body1.translation()[1], 0))
    sph1.relocate(zencad.translate(body2.translation()[0], body2.translation()[1], 0))

zencad.show(animate=animate)