
import math
from terminus.ga201.point import Point
import numpy

class Motor2:
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    @staticmethod
    def rotation(rads):
        z = math.sin(rads/2)
        w = math.cos(rads/2)
        return Motor2(0, 0, z, w)

    @staticmethod
    def translation(x, y):
        x = x/2
        y = y/2
        z = 0
        w = 1
        return Motor2(x, y, z, w)

    def __mul__(self, other):
        q = self
        p = other
        return Motor2(
            q.w*p.x + q.x*p.w - q.z*p.y + q.y*p.z,
            q.w*p.y + q.y*p.w - q.x*p.z + q.z*p.x,
            q.w*p.z + q.z*p.w,
            q.w*p.w - q.z*p.z
        )

    def transform_point(self, p):
        q = self
        return Point(
            (q.w**2 - q.z**2)*p.x - 2*q.w*q.z*p.y + (2*q.w*q.x - 2*q.z*q.y)*p.z,
            (q.w**2 - q.z**2)*p.y + 2*q.w*q.z*p.x + (2*q.w*q.y + 2*q.z*q.x)*p.z,
            (q.w**2 + q.z**2)*p.z
        )

    def transform(self, o):
        if isinstance(o, Point):
            return self.transform_point(o)

    def __repr__(self):
        return "Motor(%s, %s, %s, %s)" % (self.x, self.y, self.z, self.w)
    
    def __str__(self):
        return repr(self)

    def factorize_rotation_angle(self):
        return math.atan2(self.z, self.w) * 2

    def factorize_rotation(self):
        return Motor2(0,0,self.z,self.w)

    def reverse(self):
        return Motor2(-self.x, -self.y, -self.z, self.w)

    def inverse(self):
        rotation = self.factorize_rotation()
        translation = self.factorize_translation()
        return rotation.reverse() * translation.reverse()
        
    def factorize_translation(self):
        q = self
        return Motor2(
            q.w*q.x - q.z*q.y,
            q.w*q.y + q.z*q.x,
            0,
            1
        )

    def factorize_translation_point(self):
        #probe = Point(0,0)
        #r = self.transform_point(probe)
        #return Motor(r.x/2, r.y/2, 0, 1)
        q = self
        return Point(
            q.w*q.x - q.z*q.y,
            q.w*q.y + q.z*q.x,
            1
        )

    def factorize_translation_vector(self):
        ft = self.factorize_translation()
        return numpy.array([ft.x, ft.y])

        
    def factorize_parameters(self):
        t = self.factorize_translation()
        angle = self.factorize_rotation_angle()
        x = t.x * 2
        y = t.y * 2
        return (angle, (x,y))

    def rotate_nparray(self, a):
        angle = self.factorize_rotation_angle()
        s = math.sin(angle)
        c = math.cos(angle)
        return numpy.array([ 
            c*a[0] - s*a[1],
            s*a[0] + c*a[1]
        ])

    def rotate_nparray_inverse(self, a):
        angle = self.factorize_rotation_angle()
        s = math.sin(angle)
        c = math.cos(angle)
        return numpy.array([ 
            c*a[0] + s*a[1],
            -s*a[0] + c*a[1]
        ])

    def __eq__(self, oth):
        return (
            self.x == oth.x and
            self.y == oth.y and
            self.z == oth.z and
            self.w == oth.w
        )

    def __sub__(self, oth):
        return Motor2(
            self.x - oth.x,
            self.y - oth.y,
            self.z - oth.z,
            self.w - oth.w
        )

    def __truediv__(self, s):
        return Motor2(
            self.x / s,
            self.y / s,
            self.z / s,
            self.w / s
        )

    def is_zero_equal(self, eps = 1e-8):
        a = numpy.array([self.x, self.y, self.z, self.w])
        return numpy.linalg.norm(a) < eps