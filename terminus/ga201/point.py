
class Point201:
    def __init__(self, x, y, z=1):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return str((self.x, self.y, self.z))