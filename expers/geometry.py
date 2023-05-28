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

ip = draw_point2(Point(0,0))

q1 = Motor.rotation(0)
q2 = Motor.translation(2,0)
q3 = Motor.rotation(0)
q4 = Motor.translation(2,0)
q = q1 * q2 * q3 * q4
tr=q.factorize_translation()

#def animate(wdg):
#    q1 = Motor.rotation(0)
#    q2 = Motor.translation(2,0)
#    q3 = Motor.rotation(0)
#    q4 = Motor.translation(2,0)
#    q = q1 * q2 * q3 * q4
#    tr=q.factorize_translation()

print(tr)
#zencad.show(animate=animate)
