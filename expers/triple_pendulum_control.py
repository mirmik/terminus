#!/usr/bin/env python3

#import time
from terminus.physics.indexed_matrix import IndexedVector
from terminus.physics.body import Body2
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
body3 = Body2()

world = World()
world.set_gravity(Screw2(v=[0, -10]))
#world.set_correction_enabled(False)
world.add_body(body1)
world.add_body(body2)
world.add_body(body3)

body1.set_resistance_coefficient(0.5)
body2.set_resistance_coefficient(0.5)
body3.set_resistance_coefficient(0.5)

body1.set_position(Motor2.translation(0, -15))
body2.set_position(Motor2.translation(10, -10) * Motor2.rotation(math.pi/4))
body3.set_position(Motor2.translation(20, -10) * Motor2.rotation(math.pi/4))

force_link1 = VariableMultiForce(child=body1, parent=None, position=Motor2.translation(0, 0), senses=[
    Screw2(v=[1, 0]), Screw2(v=[0, 1])], stiffness=[1, 1])
world.add_link_force(force_link1)

force_link2 = VariableMultiForce(child=body2, parent=body1, position=Motor2.translation(0, -15), senses=[
    Screw2(v=[1, 0]), Screw2(v=[0, 1])], stiffness=[1, 1])
world.add_link_force(force_link2)

force_link3 = VariableMultiForce(child=body3, parent=body2, position=Motor2.translation(10, -10), senses=[
    Screw2(v=[1, 0]), Screw2(v=[0, 1])], stiffness=[1, 1])
world.add_link_force(force_link3)

ctrlink1 = ControlLink(position=Motor2.translation(0, 0), 
    child=body1, parent=None, senses=[Screw2(m=1)])

ctrlink2 = ControlLink(position=Motor2.translation(0, -15),
    child=body2, parent=body1, senses=[Screw2(m=1)])

ctrlink3 = ControlLink(position=Motor2.translation(10, -10),
    child=body3, parent=body2, senses=[Screw2(m=1)])

# ctrframe1 = ControlTaskFrame(
#     linked_body=body1, 
#     position_in_body=Motor2.translation(0, 0))
# ctrframe1.add_control_frame(ctrlink1)

ctrframe2 = ControlTaskFrame(
    linked_body=body2, 
    position_in_body=Motor2.translation(0, 0))
ctrframe2.add_control_frame(ctrlink1)
ctrframe2.add_control_frame(ctrlink2)

ctrframe3 = ControlTaskFrame(
    linked_body=body3, 
    position_in_body=Motor2.translation(0, 0))
ctrframe3.add_control_frame(ctrlink1)
ctrframe3.add_control_frame(ctrlink2)
ctrframe3.add_control_frame(ctrlink3)

world.add_control_link(ctrlink1)
world.add_control_link(ctrlink2)
world.add_control_link(ctrlink3)
# world.add_control_task_frame(ctrframe1)
world.add_control_task_frame(ctrframe2)
world.add_control_task_frame(ctrframe3)



sph0 = zencad.disp(zencad.sphere(1))
sph1 = zencad.disp(zencad.sphere(1))
sph2 = zencad.disp(zencad.sphere(1))

tsph = zencad.disp(zencad.sphere(1))
tsph.set_color(zencad.Color(0,1,0))

#start_time = time.time()
DDD = 1000

def control3(delta):
        current_vel = ctrframe3.right_velocity_global()
        curpos = ctrframe3.position()
        curpos = curpos.factorize_translation_vector()

        curtime = world.time()
        ctrframe3.curtime += delta

        #print(world.outkernel_operator(ctrframe))

        D = 1
        s = (math.sin((curtime)/D))
        c = (math.cos((curtime)/D))

        ds = (math.cos((curtime)/D))/D
        dc = -(math.sin((curtime)/D))/D

        d2s = -(math.sin((curtime)/D))/D/D
        d2c = -(math.cos((curtime)/D))/D/D
        
        A = 6
        B = 6

        target_pos = (numpy.array([20,5]) 
            + (s) * numpy.array([A,0])
            + (c) * numpy.array([0,B])
        )
        target_vel =( (ds) * numpy.array([A,0])
            + (dc) * numpy.array([0,B])
        )
        target_acc =( (d2s) * numpy.array([A,0])
            + (d2c) * numpy.array([0,B]))

        errorpos = Screw2(v=target_pos - curpos)
        if errorpos.norm() < 5:
            control_spd = errorpos * 8
        else:
            control_spd = errorpos * 1
        errorspd = (control_spd - current_vel)
        if errorpos.norm() < 5:
            #y = 1 - (errorpos.norm() / 4)
            errorspd = errorspd #+ Screw2(v=target_vel) #* y
        erroracc = errorspd * 400 
        if errorpos.norm() < 5:
            #y = 1 - (errorpos.norm() / 4)
            erroracc = erroracc #+ Screw2(v=target_acc) #* y
       
        norm = erroracc.norm()
        if norm > DDD:
            erroracc = erroracc * (DDD / norm)

        erroracc = erroracc * 1
        return erroracc, target_pos


def control2(delta):
        current_vel = ctrframe2.right_velocity_global()
        curpos = ctrframe2.position()
        curpos = curpos.factorize_translation_vector()
        target_pos = numpy.array([15,-10])
        target_vel = numpy.array([0,0])
        target_acc = numpy.array([0,0])
        errorpos = Screw2(v=target_pos - curpos)
        control_spd = errorpos * 2
        errorspd = (control_spd - current_vel)
        if errorpos.norm() < 5:
            errorspd = errorspd #+ Screw2(v=target_vel)
        erroracc = errorspd * 320 
        if errorpos.norm() < 5:
            erroracc = erroracc #+ Screw2(v=target_acc)
        norm = erroracc.norm()
        if norm > DDD:
            erroracc = erroracc * (DDD / norm)
        return erroracc 

# def control1(delta):
#         current_vel = ctrframe1.right_velocity_global()
#         control_spd = Screw2(v=[1,-1])
#         errorspd = (control_spd - current_vel)
#         erroracc = errorspd * 100        
#         norm = erroracc.norm()
#         if norm > DDD:
#             erroracc = erroracc * (DDD / norm)
#         return erroracc 

#while True:
def animate(wdg):
    f1 = ctrframe3.derivative_by_frame(ctrlink1)
    f2 = ctrframe3.derivative_by_frame(ctrlink2)
    f3 = ctrframe3.derivative_by_frame(ctrlink3)
    m1 = f1.matrix
    m2 = f2.matrix
    m3 = f3.matrix
    m = numpy.concatenate((m1,m2,m3), axis=1)
    JJ = numpy.linalg.pinv(m) @ m
    JJJ = numpy.eye(3) - JJ

    f1 = ctrframe2.derivative_by_frame(ctrlink1)
    f2 = ctrframe2.derivative_by_frame(ctrlink2)
    m1 = f1.matrix
    m2 = f2.matrix
    m3 = numpy.zeros((2,1))
    m = numpy.concatenate((m1,m2,m3), axis=1)
    JJ2 = numpy.linalg.pinv(m) @ m
    JJJ2 = numpy.eye(3) - JJ2

    ctrframe3.set_filter(JJ)
    ctrframe2.set_filter(JJJ)
    # ctrframe1.set_filter(JJJ @ JJJ2)

    ctr3, ctrpos = control3(0.02)
    ctr2 = control2(0.02)
    # ctr1 = control1(0.02)
    # ctrframe1.set_control_screw(ctr1)
    ctrframe2.set_control_screw(ctr2)
    ctrframe3.set_control_screw(ctr3)
    world.iteration(0.02)

    publisher.publish("pendulum/torque", world.last_solution()[2].matrix)

    sph0.relocate(zencad.translate(body1.translation()[0], body1.translation()[1], 0))
    sph1.relocate(zencad.translate(body2.translation()[0], body2.translation()[1], 0))
    sph2.relocate(zencad.translate(body3.translation()[0], body3.translation()[1], 0))

    tsph.relocate(zencad.translate(ctrpos[0], ctrpos[1], 0))

zencad.show(animate=animate, animate_step=0.02)