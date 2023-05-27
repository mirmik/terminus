#!/usr/bin/env python3

from terminus.zencad_adapter import draw_line2, draw_point2, draw_line2_positive
from terminus.ga201.line import Line
from terminus.ga201.point import Point
from terminus.ga201.join import join_point_point, oriented_distance

import zencad

p = Point(1, 1)
q = Point(0, 1)
r = Point(0, 0)

draw_point2(p)
draw_point2(q)
draw_point2(r)

lines = [
    join_point_point(q, p),
    join_point_point(r, q),
    join_point_point(p, r)
]

for line in lines:
    draw_line2(line)
    draw_line2_positive(line, 0.1, 0.05)

k = Point(0.5, 0.7)
draw_point2(k)
for i in range(3):
    print(oriented_distance(k, lines[i]).unitize())

zencad.show()
