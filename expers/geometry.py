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

class SensorPoint(zencad.assemble.unit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add(zencad.point3(0,0))

    def evaluate_parent_kinematic_units(self):
        self.parent_kinematic_units = []
        for p in self.parents_list():
            if isinstance(p, zencad.assemble.kinematic_unit):
                self.parent_kinematic_units.append(p)



class Link(zencad.assemble.unit):
    def __init__(self, l):
        super().__init__()
        self.add(zencad.segment((0,0),(0,l)))
        self.spoints = [
            SensorPoint(location=zencad.move(0,l/2,0)),
            SensorPoint(location=zencad.move(0,l,0)),
        ]
        for s in self.spoints:
            self.add_child(s)

class Manipulator(zencad.assemble.unit):
    def __init__(self):
        super().__init__()
        self.N = 4
        self.L = 1
        self.make_body()
        self.sensor_points = self.get_sensor_points()
        for s in self.sensor_points:
            s.evaluate_parent_kinematic_units()

        for s in self.sensor_points:
            print(s.parent_kinematic_units)

    def get_sensor_points(self):
        spoints = []
        for u in self.links:
            spoints.extend(u.spoints)
        return spoints

    def make_body(self):
        self.links = [Link(self.L) for i in range(self.N)]
        self.rots = [zencad.assemble.rotator((0,0,1))  for i in range(self.N)]

        for u, r in zip(self.links, self.rots):
            r.link(u)

        self.add_child(self.rots[0])
        for i in range(self.N-1):
            self.links[i].add_child(self.rots[i+1])
            self.rots[i+1].relocate(zencad.move(0,self.L,0))
        
        for i in range(self.N):
            self.rots[i].set_coord(-math.pi/8)

    #def barier_velocities_for_body(self, body):

    

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

manipulator = Manipulator()

zencad.disp(manipulator)

body = ConvexBody.from_points([
    Point(1,0),
    Point(1,1),
    Point(2,0),
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
