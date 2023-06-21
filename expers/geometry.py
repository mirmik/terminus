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

class Jacobian:
    def __init__(self, matrix):
        self.matrix = matrix

    def __str__(self):
        return str(self.matrix)

    def solve(self, vector):
        p = numpy.linalg.pinv(self.matrix)
        return p.dot(vector)

    def dot(self, vector):
        return self.matrix.dot(vector)

class SensorPoint(zencad.assemble.unit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add(zencad.point3(0,0))

    def evaluate_parent_kinematic_units(self):
        self.parent_kinematic_units = []
        for p in self.parents_list():
            if isinstance(p, zencad.assemble.kinematic_unit):
                self.parent_kinematic_units.append(p)
        self.parent_kinematic_units = list(reversed(self.parent_kinematic_units))

    def jacobi_matrix(self):
        return Jacobian(right_jacobi_matrix_lin2(self.parent_kinematic_units, self))

    def draw_sensivities_arrow(self, sens):
        for p in self.parent_kinematic_units:
            sens = right_sensivity_screw2_of_kinunit(p, self)
            motor = zencad_transform_to_motor2(self.global_location)
            sens = sens.rotate_by_angle(motor.factorize_rotation_angle())
            center = self.global_location.translation()
            arrow = zencad.segment(center, center+zencad.point3(sens.v[0], sens.v[1], 0))
            zencad.display(arrow)

    def global_to_local(self, vec):
        motor = zencad_transform_to_motor2(self.global_location)
        return motor.rotate_nparray_inverse(vec)

    def local_to_global(self, vec):
        motor = zencad_transform_to_motor2(self.global_location)
        return motor.rotate_nparray(vec)


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
        self.N = 5
        self.L = 1
        self.A = -math.pi/6
        self.make_body()
        self.sensor_points = self.get_sensor_points()
        for s in self.sensor_points:
            s.evaluate_parent_kinematic_units()

        self.final_sensor = self.sensor_points[-1]

    def draw_sensivities_arrow(self):
        for s in self.sensor_points:
            s.draw_sensivities_arrow(s)

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
            self.rots[i].set_coord(self.A)

        self.location_update()

    def final_jacobian(self):
        return self.final_sensor.jacobi_matrix()

    def apply_speed_control(self, s, delta_t = 0.01):
        for i in range(self.N):
            self.rots[i].set_coord(self.rots[i].coord + s[i] * delta_t)
        self.location_update()

    

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
#manipulator.draw_sensivities_arrow()

body = ConvexBody.from_points([
    Point(1,0),
    Point(1,1),
    Point(2,0),
])

draw_body2(body)

def animate(wdg):
    t = time.time()
    j = manipulator.final_sensor.jacobi_matrix()

    current_pos = manipulator.final_sensor.global_location.translation()
    current_pos = numpy.array([current_pos.x, current_pos.y])

    if (t%3) < 1:
        target_pos = numpy.array([1,0])
    elif (t%3) < 2:
        target_pos = numpy.array([1,1])
    else:
        target_pos = numpy.array([2,0])
    #else:
    #    target_pos = numpy.array([2,0])
    
    tgt = (target_pos - current_pos)  * 5


    target_speed = manipulator.final_sensor.global_to_local(tgt)
    s = j.solve(target_speed)
    d = j.dot(s)
    t = manipulator.final_sensor.local_to_global(d)
    
    s = s.reshape((manipulator.N))
    
    manipulator.apply_speed_control(s)


zencad.show(animate = animate)
animate(None)
zencad.show()