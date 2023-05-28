
import math


class Motor:
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    @staticmethod
    def rotation(rads):
        z = math.sin(rads)
        w = math.cos(rads)
        return Motor(0, 0, z, w)

    @staticmethod
    def translation(x, y)
        x = x
        y = y
        z = 0
        w = 1

    def __mul__(self, other):
        q = self
        p = other
        return Motor(
            q.w*p.x + q.x*p.w + q.z*p.y - q.y*p.z,
            q.w*p.y + q.y*p.w + q.x*p.z - q.z*p.x,
            q.w*q.z + q.z*p.w,
            q.w * p.w - q.z * p.z
        )