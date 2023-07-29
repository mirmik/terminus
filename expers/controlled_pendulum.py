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
import math
from terminus.physics.control_link import ControlLink
from terminus.physics.control_link import ControlTaskFrame

import zencad

import rxsignal.rxmqtt 

publisher = rxsignal.rxmqtt.mqtt_rxclient()

#numpy.set_printoptions(precision=1, suppress=True)

body1 = Body2(mass=1)
body2 = Body2()
#body3 = Body2()

world = World()
world.set_gravity(Screw2(v=[0, -10]))
world.add_body(body1)
world.add_body(body2)
#world.add_body(body3)

body1.set_resistance_coefficient(0.1)
body2.set_resistance_coefficient(0.1)
#body3.set_resistance_coefficient(5)

body1.set_position(Motor2.translation(0, -10))
body2.set_position(Motor2.translation(10, -10) * Motor2.rotation(math.pi/4))
#body3.set_position(Motor2.translation(20, -10) * Motor2.rotation(math.pi/4))

force_link1 = VariableMultiForce(child=body1, parent=None, position=Motor2.translation(0, 0), senses=[
    Screw2(v=[1, 0]), Screw2(v=[0, 1])], stiffness=[1, 1], use_child_frame=False)
world.add_link_force(force_link1)

force_link2 = VariableMultiForce(child=body2, parent=body1, position=Motor2.translation(0, -10), senses=[
    Screw2(v=[1, 0]), Screw2(v=[0, 1])], stiffness=[1, 1], use_child_frame=False)
world.add_link_force(force_link2)

#force_link3 = VariableMultiForce(child=body3, parent=body2, position=Motor2.translation(10, -10), senses=[
#    Screw2(v=[1, 0]), Screw2(v=[0, 1])], stiffness=[5, 5], use_child_frame=False)
#world.add_link_force(force_link3)

ctrlink1 = ControlLink(position=Motor2.translation(0, 0), 
    child=body1, parent=None, senses=[Screw2(m=1)], 
    use_child_frame=False)

ctrlink2 = ControlLink(position=Motor2.translation(0, -10),
    child=body2, parent=body1, senses=[Screw2(m=1)], 
    use_child_frame=False)

#ctrlink3 = ControlLink(position=Motor2.translation(10, -10),
#    child=body3, parent=body2, senses=[Screw2(m=1)], 
#    use_child_frame=False)

ctrframe = ControlTaskFrame(
    linked_body=body2, 
    position_in_body=Motor2.translation(0, 0))

world.add_control_link(ctrlink1)
world.add_control_link(ctrlink2)
#world.add_control_link(ctrlink3)
world.add_control_task_frame(ctrframe)

sph0 = zencad.disp(zencad.sphere(1))
sph1 = zencad.disp(zencad.sphere(1))
#sph2 = zencad.disp(zencad.sphere(1))

tsph = zencad.disp(zencad.sphere(1))

start_time = time.time()
planned_time = start_time

#while True:
def animate(wdg):
    global planned_time
    current_time = time.time()
    world.iteration(0.01)
    world.correction()

    print()
    #print(world.last_solution()[2])

    publisher.publish("pendulum/torque", world.last_solution()[2].matrix)

    sph0.relocate(zencad.translate(body1.translation()[0], body1.translation()[1], 0))
    sph1.relocate(zencad.translate(body2.translation()[0], body2.translation()[1], 0))
    #sph2.relocate(zencad.translate(body3.translation()[0], body3.translation()[1], 0))

    tsph.relocate(zencad.translate(ctrlink2.target[0], ctrlink2.target[1], 0))

while True:
    animate(None)
    print(ctrframe.current_position())
    print(body2.right_acceleration())
    print(ctrframe.right_acceleration_global())
    break
zencad.show(animate=animate)