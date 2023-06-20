#!/usr/bin/env python3

import numpy
import zencad
import zencad.assemble
import terminus.ga201.point as point
import terminus.ga201.join as join

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
    print("DRAW:", a, b)
    return zencad.display(zencad.segment(zencad.point3(a.x, a.y, 0), zencad.point3(b.x, b.y, 0)))

def draw_point2(point):
    point = point.unitized()
    return zencad.display(zencad.point3(point.x, point.y, 0))

def draw_body2(body):
    cpnts = [(p.x, p.y) for p in [p.unitized() for p in body.vertices()]]
    print(cpnts)
    c = ConvexHull(cpnts)
    zpoints = [zencad.point3(cpnts[i][0], cpnts[i][1]) for i in c.vertices]
    return zencad.display(zencad.polygon(zpoints))

    