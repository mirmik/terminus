#!/usr/bin/env python3
"""
Инерционные характеристики для 2D многотельной динамики.
"""

import numpy as np
from termin.geombase.pose2 import Pose2
from termin.geombase.pose3 import Pose3
from termin.geombase.screw import Screw2, cross2d_scalar


def skew(v: np.ndarray) -> np.ndarray:
    """Возвращает кососимметричную матрицу для вектора v."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


class SpatialInertia2D:
    """
    Пространственная инерция для 2D тела.
    mass: масса тела
    inertia_com: момент инерции относительно центра масс
    com: 2-вектор центра масс в локальной системе
    """
    def __init__(self, mass: float= 0.0, inertia: float= 0.0, com: np.ndarray = np.zeros(2)):
        self.m = float(mass)
        self.c = np.asarray(com).reshape(2)
        self.I_com = float(inertia)
        
        self.inertia = self.I_com
        self.mass = self.m
        self.center_of_mass = self.c

    def transform_by(self, pose: Pose2) -> "SpatialInertia2D":
        """
        Трансформировать инерцию в новую систему координат.
        В 2D момент инерции инвариантен относительно поворота.
        Центр масс переводится напрямую.
        """
        c_new = pose.transform_point(self.c)
        return SpatialInertia2D(self.m, self.I_com, c_new)

    def __add__(self, other):
        """
        Сумма двух spatial inertia в одном фрейме.
        """
        if not isinstance(other, SpatialInertia2D):
            return NotImplemented
        m1, m2 = self.m, other.m
        c1, c2 = self.c, other.c
        I1, I2 = self.I_com, other.I_com
        m = m1 + m2
        if m == 0.0:
            return SpatialInertia2D(0.0, 0.0, np.zeros(2))
        c = (m1 * c1 + m2 * c2) / m
        d1 = c1 - c
        d2 = c2 - c
        # В 2D Штейнер: I = I1 + I2 + m1*||d1||^2 + m2*||d2||^2
        I = I1 + I2 + m1 * np.dot(d1, d1) + m2 * np.dot(d2, d2)
        return SpatialInertia2D(m, I, c)

    def to_matrix(self):
        """
        Вернуть 3x3 матрицу пространственной инерции для 2D:
        | I   -m*cy  m*cx |
        | m*cy  m    0   |
        | -m*cx 0    m   |
        """
        cx, cy = self.c
        m = self.m
        I = self.I_com + m * (cx**2 + cy**2)
        # Формируем матрицу [moment, force_x, force_y]
        return np.array([
            [I,   -m*cy,  m*cx],
            [m*cy,   m,     0],
            [-m*cx,  0,     m]
        ])

    def get_kinetic_energy(self, velocity: np.ndarray, omega: float) -> float:
        v_squared = np.dot(velocity, velocity)
        return 0.5 * self.m * v_squared + 0.5 * self.I_com * omega**2

    def gravity_wrench(self, inertia_pose: Pose2, gravity: np.ndarray) -> Screw2:
        F_gravity = self.m * gravity
        cm_global = inertia_pose.transform_point(self.c)
        r_cm = cm_global - inertia_pose.lin
        torque = cross2d_scalar(r_cm, F_gravity)
        return Screw2(ang=np.array([torque]), lin=F_gravity)

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