#!/usr/bin/env python3
"""
Инерционные характеристики для 3D многотельной динамики.
"""

import numpy as np
from termin.geombase.pose3 import Pose3
from termin.geombase.screw import Screw3

def skew(v: np.ndarray) -> np.ndarray:
    """Возвращает кососимметричную матрицу для вектора v."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

class SpatialInertia3D:
    def __init__(self, mass, inertia, com: np.ndarray = np.zeros(3)):
        """
        mass         : масса тела
        com          : 3-вектор центра масс в локальной системе
        inertia (com)  : 3×3 тензор инерции относительно центра масс (в локальной системе)
        """
        self.m = float(mass)
        self.c = np.asarray(com).reshape(3)
        self.I_com = np.asarray(inertia).reshape(3, 3)

    def transform_by(self, pose : Pose3) -> "SpatialInertia3D":
        # новый центр масс в новом фрейме
        c_new = pose.transform_point(self.c)

        # поворот тензора из локального фрейма в мировой
        R = pose.rotation_matrix()
        I_rot = R @ self.I_com @ R.T

        # перенос оси из центра масс в новый фрейм
        c_skew = skew(c_new)
        I_new = I_rot + self.m * c_skew @ c_skew.T

        return SpatialInertia3D(self.m, I_new, c_new)

    def __add__(self, other):
        """Сумма двух spatial inertia в одном фрейме."""
        if not isinstance(other, SpatialInertia3D):
            return NotImplemented

        m1, m2 = self.m, other.m
        c1, c2 = self.c, other.c
        I1, I2 = self.I_com, other.I_com

        # общая масса
        m = m1 + m2
        if m == 0.0:
            return SpatialInertia3D(0.0, np.zeros((3,3)), np.zeros(3))

        # общий центр масс
        c = (m1 * c1 + m2 * c2) / m

        # смещения от индивидуальных COM до общего
        d1 = c1 - c
        d2 = c2 - c

        # перенос инерций к общему центру масс
        I = (
            I1 + I2
            + m1 * skew(d1) @ skew(d1).T
            + m2 * skew(d2) @ skew(d2).T
        )

        return SpatialInertia3D(m, I, c)


    def to_matrix(self):
        c_skew = skew(self.c)
        upper_left  = self.I_com + self.m * (c_skew @ c_skew.T)
        upper_right = self.m * c_skew
        lower_left  = -self.m * c_skew.T
        lower_right = self.m * np.eye(3)
        return np.block([
            [upper_left,  upper_right],
            [lower_left,  lower_right]
        ])

    def to_matrix_vw_order(self):
        upper_left  = self.I_com + self.m * (skew(self.c) @ skew(self.c).T)
        upper_right = self.m * skew(self.c)
        lower_left  = -self.m * skew(self.c).T
        lower_right = self.m * np.eye(3)
        return np.block([
            [lower_right,  lower_left],
            [upper_right,  upper_left]
        ])
        
    def gravity_wrench(self, inertia_pose: Pose3, gravity: np.ndarray) -> Screw3:
        """
        Вернуть гравитационный винт (сила и момент).
        """
        spatial_world = self.transform_by(inertia_pose)
        m = spatial_world.m
        c_world = spatial_world.c          # COM в мире
        Fg = m * gravity                   # линейная сила
        tau_g = np.cross(c_world, Fg)      # момент от тяжести

        return Screw3(ang=tau_g, lin=Fg)