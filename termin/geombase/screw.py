import numpy
import math
from .pose3 import Pose3

class Screw:
    """A class representing a pair of vector and bivector"""
    def __init__(self, ang, lin):
        self.ang = ang  # Bivector part
        self.lin = lin  # Vector part

        if not isinstance(self.ang, numpy.ndarray):
            raise Exception("ang must be ndarray")

        if not isinstance(self.lin, numpy.ndarray):
            raise Exception("lin must be ndarray")

    def __repr__(self):
        return f"Screw(ang={self.ang}, lin={self.lin})"

class Screw2(Screw):
    """A 2D Screw specialized for planar motions."""
    def __init__(self, ang:float, lin: numpy.ndarray):
        super().__init__(ang=ang, lin=lin)

    def moment(self) -> float:
        """Return the moment (bivector part) of the screw."""
        return self.ang

    def vector(self) -> numpy.ndarray:
        """Return the vector part of the screw."""
        return self.lin

    def kinematic_carry(self, arm: "Vector2") -> "Screw2":
        """Carry the screw by arm. For pair of angular and linear speeds."""
        return Screw2(
            lin=self.lin + self.ang * numpy.array([-arm[1], arm[0]]),
            ang=self.ang)

    def force_carry(self, arm: "Vector2") -> "Screw2":
        """Carry the screw by arm. For pair of torques and forces."""
        return Screw2(
            ang=self.ang - (arm[0]*self.lin[1] - arm[1]*self.lin[0]),
            lin=self.lin)

class Screw3(Screw):
    """A 3D Screw specialized for spatial motions."""
    def __init__(self, ang: numpy.ndarray = numpy.array([0,0,0]), lin: numpy.ndarray = numpy.array([0,0,0])):
        super().__init__(ang=ang, lin=lin)

    def moment(self) -> numpy.ndarray:
        """Return the moment (bivector part) of the screw."""
        return self.ang

    def vector(self) -> numpy.ndarray:
        """Return the vector part of the screw."""
        return self.lin

    def kinematic_carry(self, arm: "Vector3") -> "Screw3":
        """Twist transform. Carry the screw by arm. For pair of angular and linear speeds."""
        return Screw3(
            lin=self.lin + numpy.cross(self.ang, arm),
            ang=self.ang)

    def force_carry(self, arm: "Vector3") -> "Screw3":
        """Wrench transform. Carry the screw by arm. For pair of torques and forces."""
        return Screw3(
            ang=self.ang - numpy.cross(arm, self.lin),
            lin=self.lin)

    def twist_carry(self, arm: "Vector3") -> "Screw3":
        """Alias for kinematic_carry."""
        return self.kinematic_carry(arm)

    def wrench_carry(self, arm: "Vector3") -> "Screw3":
        """Alias for force_carry."""
        return self.force_carry(arm)

    def transform_by(self, trans):
        return Screw3(
            ang=trans.transform_vector(self.ang),
            lin=trans.transform_vector(self.lin)
        )

    def inverse_transform_by(self, trans):
        return Screw3(
            ang=trans.inverse_transform_vector(self.ang),
            lin=trans.inverse_transform_vector(self.lin)
        )

    def transform_as_twist_by(self, trans):
        return Screw3(
            ang = trans.transform_vector(self.ang),
            lin = trans.transform_vector(self.lin + numpy.cross(trans.lin, self.ang))
        )

    def inverse_transform_as_twist_by(self, trans):
        return Screw3(
            ang = trans.inverse_transform_vector(self.ang),
            lin = trans.inverse_transform_vector(self.lin - numpy.cross(trans.lin, self.ang))
        )

    def transform_as_wrench_by(self, trans):
        """Transform wrench (moment + force) under SE(3) transform."""
        p = trans.lin
        return Screw3(
            ang = trans.transform_vector(self.ang + numpy.cross(p, self.lin)),
            lin = trans.transform_vector(self.lin)
        )

    def inverse_transform_as_wrench_by(self, trans):
        """Inverse transform of a wrench under SE(3) transform."""
        p = trans.lin
        return Screw3(
            ang = trans.inverse_transform_vector(self.ang - numpy.cross(p, self.lin)),
            lin = trans.inverse_transform_vector(self.lin)
        )

    def as_pose3(self):
        """Convert the screw to a Pose3 representation (for small motions)."""
        rotangle = numpy.linalg.norm(self.ang)
        if rotangle < 1e-8:
            # Pure translation
            return Pose3(
                ang=numpy.array([0.0, 0.0, 0.0, 1.0]),
                lin=self.lin
            )
        axis = self.ang / rotangle
        half_angle = rotangle / 2.0
        q = numpy.array([
            axis[0] * math.sin(half_angle),
            axis[1] * math.sin(half_angle),
            axis[2] * math.sin(half_angle),
            math.cos(half_angle)
        ])
        return Pose3(
            ang=q,
            lin=self.lin
        )

    def __mul__(self, oth):
        return Screw3(self.ang * oth, self.lin * oth)