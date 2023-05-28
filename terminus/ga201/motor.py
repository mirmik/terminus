
import math

from terminus.ga201.point import Point

class Motor:
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    @staticmethod
    def rotation(rads):
        z = math.sin(rads/2)
        w = math.cos(rads/2)
        return Motor(0, 0, z, w)

    @staticmethod
    def translation(x, y):
        x = x/2
        y = y/2
        z = 0
        w = 1
        return Motor(x, y, z, w)

    def __mul__(self, other):
        q = self
        p = other
        return Motor(
            q.w*p.x + q.x*p.w - q.z*p.y + q.y*p.z,
            q.w*p.y + q.y*p.w - q.x*p.z + q.z*p.x,
            q.w*p.z + q.z*p.w,
            q.w*p.w - q.z*p.z
        )

    def transform_point(self, p):
        q = self
        a = q.w
        a12 = q.z
        a31 = q.y
        a23 = q.x
        return Point(
            (a**2 - a12**2)*p.x - 2*a*a12*p.y + (2*a*a23 - 2*a12*a31)*p.z,
            (a**2 - a12**2)*p.y + 2*a*a12*p.x + (2*a*a31 + 2*a12*a23)*p.z,
            (a**2 + a12**2)*p.z
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
        return Motor(0,0,self.z,self.w)

    def reverse(self):
        return Motor(-self.x, -self.y, -self.z, self.w)

    def factorize_translation(self):
        factorize_rotation = self.factorize_rotation()
        inv_rotation = factorize_rotation.reverse()
        inverted = inv_rotation * self
        p = Point(inverted.x, inverted.y, 1)
        r = factorize_rotation.transform_point(p)
        return Motor(r.x, r.y, 0, 1)
        
    def factorize_parameters(self):
        t = self.factorize_translation()
        angle = self.factorize_rotation_angle()
        x = t.x * 2
        y = t.y * 2
        return (angle, (x,y))