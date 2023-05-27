#!/usr/bin/env python3

from point import Point201
from line import Line201

def join201_point_point(p, q):
    return Line201(
        p.y*q.z - q.y*p.z,
        q.x*p.z - p.x*q.z,
        p.x*q.y - q.x*p.y
    )

def projection_point_line(p, l):
    a = (l.x*l.x + l.y*l.y)
    b = (l.x*p.x + l.y*p.y + l.z*p.z)
    return Point201(
        a * p.x - b * l.x,
        a * p.y - b * l.y,
        a * p.z
    )

if __name__ == "__main__":
    p = Point201(1, 1)
    q = Point201(1, 0)
    print(join201_point_point(p, q))

    l = Line201(1, 1, -1)
    print(projection_point_line(Point201(1, 1), l))