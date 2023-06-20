#!/usr/bin/env python3

import numpy
import zencad
import zencad.assemble
import terminus.ga201.point as point
import terminus.ga201.join as join
import terminus.ga201.screw as Screw

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

def left_sensivity_screw2_of_kinunit(kinunit):
    sensivity = zencad_transform_to_screw2(kinunit.sensivity())
    motor = zencad_transform_to_motor2(kinunit.global_location())
    return sensivity.kinematic_carry(motor.inverse())

def left_jacobi_matrix_velocities(kinunits):
    sensivities = [left_sensivity_screw2_of_kinunit(k) for k in kinunits]
    rows = 2
    cols = len(kinunits)
    mat = np.concatenate([(k.x, k.y) for k in sensivities])
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
