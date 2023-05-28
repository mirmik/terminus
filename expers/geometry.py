#!/usr/bin/env python3

from terminus.zencad_adapter import draw_line2, draw_point2, draw_line2_positive
from terminus.ga201.line import Line
from terminus.ga201.point import Point
from terminus.ga201.motor import Motor
from terminus.ga201.join import join_point_point, oriented_distance
from terminus.ga201.convex_hull import ConvexBody
import math

import zencad
zencad.disable_lazy()

zencad.set_default_point_color(zencad.Color(1,0,0))

r = Point(0, 0)
s = Point(1, 0)
p = Point(1, 2)
q = Point(0, 1)

lines = [
    join_point_point(s, r),
    join_point_point(p, s),
    join_point_point(q, p),
    join_point_point(r, q),
]

convex = ConvexBody(lines)


print("count_of_vertices", convex.count_of_vertices())
print("count_of_planes", convex.count_of_hyperplanes())

print("vertices", convex.vertices())
print("hyperplanes", convex.hyperplanes())

#for v in convex.vertices():
#    draw_point2(v)

p=Point(2, 3)
proj = convex.point_projection(p)
draw_point2(p)
draw_point2(proj)
print("proj", proj)

print("***")
print("***")
q1 = Motor.rotation(math.pi/2)
q2 = Motor.translation(20,0)
q3 = Motor.rotation(-math.pi/2)
q = q1*q2*q3
print("q1", q1)
print("q2", q2)
print("q", q)

p = Point(1,0)
qp = q.transform(p)

print("***")
print(qp)

zencad.show()
