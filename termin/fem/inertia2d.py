#!/usr/bin/env python3
"""
Инерционные характеристики для 2D многотельной динамики.
"""

import numpy as np
from termin.geombase.pose2 import Pose2
from termin.geombase.screw import Screw2, cross2d_scalar


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
