#!/usr/bin/env python3

import math
import numpy

class Screw2:
    def __init__(self, m=0, v=numpy.array([0, 0])):
        self._m = m
        self._v = numpy.array(v)

        if not isinstance(self._v, numpy.ndarray) and self._v.shape != (2,):
            raise Exception("Vector must be numpy.ndarray")

        if not isinstance(self._m, (int, float)):
            raise Exception("Moment must be int or float")

    def lin(self):
        return self._v

    def ang(self):
        return self._m

    def vector(self):
        return self._v

    def moment(self):
        return self._m

    def set_vector(self, v):
        self._v = v

    def set_moment(self, m):
        self._m = m

    def kinematic_carry(self, motor):
        angle = motor.factorize_rotation_angle()
        translation = motor.factorize_translation_vector()
        rotated_scr = self.rotate_by_angle(angle)
        m = rotated_scr._m
        v = rotated_scr._v
        b = translation
        a = -m
        new_m = m
        new_v = v + numpy.array([-a * b[1], a * b[0]])
        ret = Screw2(m=new_m, v=new_v)
        return ret

    def fulldot(self, other):
        return self._m * other._m + self._v.dot(other._v)

    def force_carry(self, motor):
        angle = motor.factorize_rotation_angle()
        translation = motor.factorize_translation_vector()
        rotated_scr = self.rotate_by_angle(angle)
        m = rotated_scr.moment()
        v = rotated_scr.vector()
        b = translation
        a = -m

        print("TODO: force carry")
        new_m = m
        new_v = v
        ret = Screw2(m=new_m, v=new_v)
        return ret

    def inverted_kinematic_carry(self, motor):
        inverted = motor.inverse()
        return self.kinematic_carry(inverted)

    def kinematic_carry_vec(self, translation):
        m = self._m
        v = self._v
        b = translation
        a = -m  # (w+v)'=w+v-w*t : из уравнения (v+w)'=(1+t/2)(v+w)(1-t/2)
        ret = Screw2(m=m, v=v + numpy.array([
            -a * b[1], a * b[0]
        ]))
        return ret

    def rotate_by_angle(self, angle):
        m = self._m
        v = self._v
        s = math.sin(angle)
        c = math.cos(angle)
        return Screw2(m=m, v=numpy.array([
            c*v[0] - s*v[1],
            s*v[0] + c*v[1]
        ]))

    def rotate_by(self, motor):
        return self.rotate_by_angle(motor.angle())

    def inverse_rotate_by(self, motor):
        return self.rotate_by_angle(-motor.angle())

    def __str__(self):
        return "Screw2(%s, %s)" % (self._m, self._v)

    def __mul__(self, s):
        return Screw2(v=self._v*s, m=self._m*s)

    def __truediv__(self, s):
        return Screw2(v=self._v/s, m=self._m/s)

    def __add__(self, oth):
        return Screw2(v=self._v+oth._v, m=self._m+oth._m)

    def toarray(self):
        return numpy.array([self.moment(), *self.vector()])

    @staticmethod
    def from_array(arr):
        return Screw2(m=arr[0], v=arr[1:])


if __name__ == "__main__":
    from terminus.ga201.motor import Motor2
    # scr = Screw2(m=0, v=[1, 0])
    # mot = Motor2.rotation(math.pi/2)
    # print(scr.kinematic_carry(mot))
    # print(scr.inverted_kinematic_carry(mot))

    scr = Screw2(m=1, v=[0, 0])
    mot = Motor2.translation(1, 0)
    print(scr.kinematic_carry(mot))
    print(scr.inverted_kinematic_carry(mot))

    # scr = Screw2(m=0, v=[1, 0])
    # print(scr.rotate_by_angle(math.pi/2))

    # scr = Screw2(m=1, v=[0, 0])
    # print(scr.rotate_by_angle(math.pi/2))
