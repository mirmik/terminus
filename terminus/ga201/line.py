import terminus.ga201.point as point
import terminus.ga201.join as join


class Line:
    def __init__(self, x, y, z):
        n = (x*x + y*y)**0.5
        self.x = x / n
        self.y = y / n
        self.z = z * n
        
    def __str__(self):
        return str((self.x, self.y, self.z))

    def parameter_point(self, t):
        dir_y = self.x
        dir_x = -self.y
        origin = point.origin()
        c = join.projection_point_line(origin, self)
        return point.Point(
            c.x + dir_x * t,
            c.y + dir_y * t
        )
