#!/usr/bin/env python3
import time

from terminus.zencad_adapter import *
from terminus.ga201.line import Line
from terminus.ga201.point import Point
from terminus.ga201.motor import Motor
from terminus.ga201.join import join_point_point, oriented_distance
from terminus.ga201.convex_body import ConvexBody
import math

import zencad
zencad.disable_lazy()

#zencad.set_default_point_color(zencad.Color(1,0,0))

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

#convex = ConvexBody(lines)
#draw_body2(convex)

# ip1 = draw_point2(Point(0,0))
# ip2 = draw_point2(Point(0,0))

# q1 = Motor.rotation(math.pi/2)
# q2 = Motor.translation(2,0)
# q3 = Motor.rotation(math.pi/2)
# q4 = Motor.translation(2,0)
# q = q1 * q2 * q3 * q4
# p = Point(1, 0)

body = ConvexBody.from_points([
    Point(0,1),
    Point(1,0),
    Point(0,0),
])

draw_body2(body)

# def motor_to_zencad_trsf(m):
#     angle, vec = m.factorize_parameters()
#     tr = zencad.translate(vec[0], vec[1], 0)
#     rot = zencad.rotate([0,0,1], angle)
#     return tr * rot

# def animate(wdg):
#     t = time.time()
#     q1 = Motor.rotation(t)
#     q2 = Motor.translation(2,0)
#     q3 = Motor.rotation(t)
#     q4 = Motor.translation(2,0)
#     q = q1 * q2 * q3 * q4
#     qq = q1 * q2

#     par1 = q.factorize_parameters()
#     par2 = qq.factorize_parameters()

#     ip1.relocate(motor_to_zencad_trsf(q))
#     ip2.relocate(motor_to_zencad_trsf(qq))

zencad.show()
#zencad.show(animate = animate)
