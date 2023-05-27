#!/usr/bin/env python3

import numpy
import zencad
import terminus.ga201.point as point
import terminus.ga201.join as join

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
    a = line.parameter_point(-100)
    b = line.parameter_point(100)
    zencad.display(zencad.segment(zencad.point3(a.x, a.y, 0), zencad.point3(b.x, b.y, 0)))

def draw_point2(point):
    zencad.display(zencad.point3(point.x, point.y, 0))
