#!/usr/bin/env python3

import numpy
import numpy as np
import zencad
import zencad.assemble
import terminus.ga201.point as point
import terminus.ga201.join as join
import terminus.ga201.screw as screw
import terminus.ga201.motor as motor

from scipy.spatial import ConvexHull

def draw_line2_positive(line, step=1, length=0.1):
    min = -100
    max = 100
    for i in numpy.arange(min, max, step):
        x = line.x
        y = line.y
        d = point.Point(x, y, 0) * length
        a = line.parameter_point(i) + d 
        b = line.parameter_point(i)
        zencad.display(zencad.segment(zencad.point3(a.x, a.y, 0), zencad.point3(b.x, b.y, 0)))

def draw_line2(line):
    unitized_line = line.unitized()
    a = unitized_line.parameter_point(-100)
    b = unitized_line.parameter_point(100)
    return zencad.display(zencad.segment(zencad.point3(a.x, a.y, 0), zencad.point3(b.x, b.y, 0)))

def draw_point2(point):
    point = point.unitized()
    return zencad.display(zencad.point3(point.x, point.y, 0))

def draw_body2(body):
    cpnts = [(p.x, p.y) for p in [p.unitized() for p in body.vertices()]]
    c = ConvexHull(cpnts)
    zpoints = [zencad.point3(cpnts[i][0], cpnts[i][1]) for i in c.vertices]
    return zencad.display(zencad.polygon(zpoints))

def zencad_sensivity_to_screw2(sensivity):
    a = sensivity[0]
    l = sensivity[1]
    return screw.Screw2(v=numpy.array([l.x, l.y]), m=a.z)

def zencad_transform_to_motor2(transform):
    l = transform.translation()
    a = transform.rotation_quat()
    return motor.Motor(l[0], l[1], 0, 1) * motor.Motor(0, 0, a.z, a.w)

def right_sensivity_screw2_of_kinunit(kinunit, senunit):
    sensivity = zencad_sensivity_to_screw2(kinunit.sensivity())
    kinmotor = zencad_transform_to_motor2(kinunit.global_location)
    senmotor = zencad_transform_to_motor2(senunit.global_location)

    # Тут вообще-то надо винт чувствительности развернуть в систему senmotor,
    # но я пока нет примера на котором это можно проверить.

    motor = senmotor.reverse() * kinmotor
    translation = motor.factorize_translation_vector()
    carried = sensivity.kinematic_carry_vec(-translation)
    #print("carried: ", carried)
    #carried_in_global_frame = carried.rotate_by_angle(kinmotor.factorize_rotation_angle())
    #print("carried_in_global_frame: ", carried_in_global_frame)
    return carried

def right_jacobi_matrix_lin2(kinunits, senunit):
    sensivities = [right_sensivity_screw2_of_kinunit(k, senunit) for k in kinunits]
    rows = 2
    cols = len(kinunits)
    mat = np.concatenate([k.v.reshape(2, 1) for k in sensivities], axis=1)
    return mat

def solve_2d_velocity_system(velocities, alphas, matrices):
    qdim = matrices.rows()
    sdim = len(matrices)
    
    b = numpy.ndarray([0]*qdim)
    for i in range(sdim):
        b += alphas[i] * numpy.matmul(matrices[i], velocities[i])

    A = numpy.ndarray(([0]*qdim)*qdim)
    for i in range(sdim):
        A += alphas[i] * numpy.matmul(matrices[i].T, matrices[i])

    #A_pinv = A.pinv()
    return solve(A, b)
