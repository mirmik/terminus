#!/usr/bin/env python3

import time
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

body1.set_resistance_coefficient(0.1)
body2.set_resistance_coefficient(0.1)
body3.set_resistance_coefficient(0.1)

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

ctrframe1 = ControlTaskFrame(
    linked_body=body1, 
    position_in_body=Motor2.translation(0, 0))
ctrframe1.add_control_frame(ctrlink1)

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
world.add_control_task_frame(ctrframe1)
world.add_control_task_frame(ctrframe2)
world.add_control_task_frame(ctrframe3)



sph0 = zencad.disp(zencad.sphere(1))
sph1 = zencad.disp(zencad.sphere(1))
sph2 = zencad.disp(zencad.sphere(1))

tsph = zencad.disp(zencad.sphere(1))
tsph.set_color(zencad.Color(0,1,0))

start_time = time.time()
planned_time = start_time

DDD = 10

def control3(delta):
        current_vel = ctrframe3.right_velocity_global()
        curpos = ctrframe3.position()
        curpos = curpos.factorize_translation_vector()

        curtime = ctrframe3.curtime
        ctrframe3.curtime += delta

        #print(world.outkernel_operator(ctrframe))

        D = 1
        s = (math.sin((curtime - start_time)/D))
        c = (math.cos((curtime - start_time)/D))

        ds = (math.cos((curtime - start_time)/D))/D
        dc = -(math.sin((curtime - start_time)/D))/D

        d2s = -(math.sin((curtime - start_time)/D))/D/D
        d2c = -(math.cos((curtime - start_time)/D))/D/D
        
        A = 2
        B = 2

        target_pos = (numpy.array([10,5]) 
            + (s) * numpy.array([A,0])
            + (c) * numpy.array([0,B])
        )
        target_vel =( (ds) * numpy.array([A,0])
            + (dc) * numpy.array([0,B])
        )
        target_acc =( (d2s) * numpy.array([A,0])
            + (d2c) * numpy.array([0,B]))

        k = curtime / 10

        errorpos = Screw2(v=target_pos - curpos)
        control_spd = errorpos * 8 + Screw2(v=target_vel)
        errorspd = (control_spd - current_vel)

        #errorspd = Screw2(v=[0,5]) - current_vel
        erroracc = errorspd * 120 +  Screw2(v=target_acc)
       
        norm = erroracc.norm()
        if norm > DDD:
            erroracc = erroracc * (DDD / norm)

        erroracc = erroracc * 1
        return erroracc, target_pos


def control2(delta):
        current_vel = ctrframe2.right_velocity_global()
        curpos = ctrframe2.position()
        curpos = curpos.factorize_translation_vector()

        curtime = ctrframe2.curtime
        ctrframe2.curtime += delta

        control_spd = Screw2(v=[1,-1])
        errorspd = (control_spd - current_vel)
        erroracc = errorspd * 1

        #print(world.kernel_operator(ctrframe2))
        f1 = ctrframe3.derivative_by_frame(ctrlink1)
        f2 = ctrframe3.derivative_by_frame(ctrlink2)
        f3 = ctrframe3.derivative_by_frame(ctrlink3)
        m1 = f1.matrix
        m2 = f2.matrix
        m3 = f3.matrix
        m = numpy.concatenate((m1,m2,m3), axis=1)
        #print(m)
        JJ = numpy.linalg.pinv(m) @ m
        JJJ = numpy.eye(3) - JJ
        print(JJJ)

        ctrframe2.set_filter(JJJ)

        norm = erroracc.norm()
        if norm > DDD:
            erroracc = erroracc * (DDD / norm)

        return erroracc 

def control1(delta):
        current_vel = ctrframe1.right_velocity_global()
        curpos = ctrframe1.position()
        curpos = curpos.factorize_translation_vector()

        curtime = ctrframe1.curtime
        ctrframe1.curtime += delta

        control_spd = Screw2(v=[-1,-1])
        errorspd = (control_spd - current_vel)
        erroracc = errorspd * 2

        norm = erroracc.norm()
        
        norm = erroracc.norm()
        if norm > DDD:
            erroracc = erroracc * (DDD / norm)

        return erroracc 

#while True:
def animate(wdg):
    global planned_time
    current_time = time.time()

    ctr3, ctrpos = control3(0.02)
    ctr2 = control2(0.02)
    ctr1 = control1(0.02)
    ctrframe1.set_control_screw(ctr1)
    ctrframe2.set_control_screw(ctr2)
    ctrframe3.set_control_screw(ctr3)
    world.iteration(0.02)

    publisher.publish("pendulum/torque", world.last_solution()[2].matrix)

    sph0.relocate(zencad.translate(body1.translation()[0], body1.translation()[1], 0))
    sph1.relocate(zencad.translate(body2.translation()[0], body2.translation()[1], 0))
    sph2.relocate(zencad.translate(body3.translation()[0], body3.translation()[1], 0))

    tsph.relocate(zencad.translate(ctrpos[0], ctrpos[1], 0))

zencad.show(animate=animate, animate_step=0.02)