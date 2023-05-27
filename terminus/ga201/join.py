#!/usr/bin/env python3

from terminus.ga201.point import Point
from terminus.ga201.line import Line
from terminus.ga201.magnitude import Magnitude
import math

def join_point_point(p, q):
    return Line(
        p.y*q.z - q.y*p.z,
        q.x*p.z - p.x*q.z,
        p.x*q.y - q.x*p.y
    )

def projection_point_line(p, l):
    a = (l.x*l.x + l.y*l.y)
    b = (l.x*p.x + l.y*p.y + l.z*p.z)
    return Point(
        a * p.x - b * l.x,
        a * p.y - b * l.y,
        a * p.z
    )

def oriented_distance(p,l):
    return Magnitude(
        p.x*l.x + p.y*l.y + p.z*l.z, 
        p.z*math.sqrt(l.x*l.x + l.y*l.y + l.z*l.z))

if __name__ == "__main__":
    p = Point(1, 1)
    q = Point(1, 0)
    print(join_point_point(p, q))

    l = Line(1, 1, -1)
    print(projection_point_line(Point(1, 1), l))

