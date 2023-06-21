
#!/usr/bin/env python3

import math
import numpy

class Screw2:
    def __init__(self, m, v):
        self.m = m
        self.v = v

    def kinematic_carry(self, motor):
        angle = motor.factorize_rotation_angle()
        translation = motor.factorize_translation_vector()
        rotated_scr = self.rotate_by_angle(angle)
        m = rotated_scr.m
        v = rotated_scr.v
        b = translation
        a = -m
        ret = Screw2(m=m, v=v + numpy.array([
            -a * b[1], a * b[0]
        ]))
        return ret

    def kinematic_carry_vec(self, translation):
        m = self.m
        v = self.v
        b = translation
        a = -m # (w+v)'=w+v-w*t : из уравнения (v+w)'=(1+t/2)(v+w)(1-t/2)
        ret = Screw2(m=m, v=v + numpy.array([
            -a * b[1], a * b[0]
        ]))
        return ret

    def rotate_by_angle(self, angle):
        m = self.m
        v = self.v
        s = math.sin(angle)
        c = math.cos(angle)
        return Screw2(m=m, v= numpy.array([ 
            c*v[0] - s*v[1],
            s*v[0] + c*v[1]
        ]))

    def force_screw_transform(self, motor):
        pass

    def __str__(self):
        return "Screw2(%s, %s)" % (self.m, self.v)