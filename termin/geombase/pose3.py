import math
import numpy

from termin.util import qmul, qrot, qslerp, qinv

class Pose3:
    """A 3D Pose represented by rotation quaternion and translation vector."""

    def __init__(self, ang: numpy.ndarray = numpy.array([0.0, 0.0, 0.0, 1.0]), lin: numpy.ndarray = numpy.array([0.0, 0.0, 0.0])):
        self.ang = ang
        self.lin = lin
        self._rot_matrix = None  # Lazy computation
        self._mat = None  # Lazy computation
        self._mat34 = None  # Lazy computation

    @staticmethod
    def identity():
        return Pose3(
            ang=numpy.array([0.0, 0.0, 0.0, 1.0]),
            lin=numpy.array([0.0, 0.0, 0.0])
        )

    def as_rotation_matrix(self):
        """Get the 3x3 rotation matrix corresponding to the pose's orientation."""
        if self._rot_matrix is None:
            x, y, z, w = self.ang
            self._rot_matrix = numpy.array([
                [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
            ])
        return self._rot_matrix

    def as_matrix(self):
        """Get the 4x4 transformation matrix corresponding to the pose."""
        if self._mat is None:
            R = self.as_rotation_matrix()
            t = self.lin
            self._mat = numpy.eye(4)
            self._mat[:3, :3] = R
            self._mat[:3, 3] = t
        return self._mat

    def as_matrix34(self):
        """Get the 3x4 transformation matrix corresponding to the pose."""
        if self._mat34 is None:
            R = self.as_rotation_matrix()
            t = self.lin
            self._mat34 = numpy.zeros((3, 4))
            self._mat34[:, :3] = R
            self._mat34[:, 3] = t
        return self._mat34

    def inverse(self):
        """Compute the inverse of the pose."""
        x, y, z, w = self.ang
        tx, ty, tz = self.lin
        inv_ang = numpy.array([-x, -y, -z, w])
        inv_lin = qrot(inv_ang, -self.lin)
        return Pose3(inv_ang, inv_lin)

    def __repr__(self):
        return f"Pose3(ang={self.ang}, lin={self.lin})"

    def transform_point(self, point: numpy.ndarray) -> numpy.ndarray:
        """Transform a 3D point using the pose."""
        return qrot(self.ang, point) + self.lin

    def transform_vector(self, vector: numpy.ndarray) -> numpy.ndarray:
        """Transform a 3D vector using the pose (ignoring translation)."""
        return qrot(self.ang, vector)

    def inverse_transform_point(self, pnt):
        return qrot(qinv(self.ang), pnt - self.lin)
    
    def inverse_transform_vector(self, vec):
        return qrot(qinv(self.ang), vec)

    def __mul__(self, other):
        """Compose this pose with another pose."""
        if not isinstance(other, Pose3):
            raise TypeError("Can only multiply Pose3 with Pose3")
        q = qmul(self.ang, other.ang)
        t = self.lin + qrot(self.ang, other.lin)
        return Pose3(ang=q, lin=t)
    
    def compose(self, other: 'Pose3') -> 'Pose3':
        """Compose this pose with another pose."""
        return self * other

    @staticmethod
    def rotation(axis: numpy.ndarray, angle: float):
        """Create a rotation pose around a given axis by a given angle."""
        axis = axis / numpy.linalg.norm(axis)
        s = math.sin(angle / 2)
        c = math.cos(angle / 2)
        q = numpy.array([axis[0] * s, axis[1] * s, axis[2] * s, c])
        return Pose3(ang=q, lin=numpy.array([0.0, 0.0, 0.0]))

    @staticmethod
    def translation(x: float, y: float, z: float):
        """Create a translation pose."""
        return Pose3(ang=numpy.array([0.0, 0.0, 0.0, 1.0]), lin=numpy.array([x, y, z]))

    @staticmethod
    def rotateX(angle: float):
        """Create a rotation pose around the X axis."""
        return Pose3.rotation(numpy.array([1.0, 0.0, 0.0]), angle)

    @staticmethod
    def rotateY(angle: float):
        """Create a rotation pose around the Y axis."""
        return Pose3.rotation(numpy.array([0.0, 1.0, 0.0]), angle)
    
    @staticmethod
    def rotateZ(angle: float):
        """Create a rotation pose around the Z axis."""
        return Pose3.rotation(numpy.array([0.0, 0.0, 1.0]), angle)

    @staticmethod
    def move(dx: float, dy: float, dz: float):
        """Move the pose by given deltas in local coordinates."""
        return Pose3.translation(dx, dy, dz)

    @staticmethod
    def moveX(distance: float):
        return Pose3.move(distance, 0.0, 0.0)
    
    @staticmethod
    def moveY(distance: float):
        return Pose3.move(0.0, distance, 0.0)

    @staticmethod
    def moveZ(distance: float):
        return Pose3.move(0.0, 0.0, distance)

    @staticmethod
    def right(distance: float):
        return Pose3.move(distance, 0.0, 0.0)

    @staticmethod
    def forward(distance: float):
        return Pose3.move(0.0, distance, 0.0)

    @staticmethod
    def up(distance: float):
        return Pose3.move(0.0, 0.0, distance)


    @staticmethod
    def lerp(pose1: 'Pose3', pose2: 'Pose3', t: float) -> 'Pose3':
        """Linearly interpolate between two poses."""
        lerped_ang = qslerp(pose1.ang, pose2.ang, t)
        lerped_lin = (1 - t) * pose1.lin + t * pose2.lin
        return Pose3(ang=lerped_ang, lin=lerped_lin)