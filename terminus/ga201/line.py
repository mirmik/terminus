


class Line201:
    def __init__(self, x, y, z):
        n = (x*x + y*y)**0.5
        self.x = x / n
        self.y = y / n
        self.z = z * n
        
    def __str__(self):
        return str((self.x, self.y, self.z))