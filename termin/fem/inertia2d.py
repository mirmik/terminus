#!/usr/bin/env python3
"""
Инерционные характеристики для 2D многотельной динамики.
"""

import numpy as np
from termin.geombase.pose2 import Pose2
from termin.geombase.screw import Screw2, cross2d_scalar


import numpy as np

def skew2(v):
    """2D псевдо-skew: ω × r = [-ω*r_y, ω*r_x].
       Здесь возвращаем 2×2 матрицу для углового ω."""
    return np.array([[0, -v],
                     [v,  0]])

class SpatialInertia2D:
    def __init__(self, mass = 0.0, inertia = 0.0, com=np.zeros(2)):
        """
        mass : масса тела
        J_com : момент инерции вокруг центра масс (скаляр)
        com : 2-вектор центра масс в локальной системе
        """
        self.m = float(mass)
        self.Jc = float(inertia)
        self.c = np.asarray(com, float).reshape(2)

    @property
    def I_com(self):
        return self.Jc

    @property
    def mass(self):
        return self.m

    @property
    def inertia(self):
        return self.Jc

    @property
    def center_of_mass(self):
        return self.c

    
    def transform_by(self, pose: Pose2) -> "SpatialInertia2D":
        """
        Трансформировать инерцию в новую систему координат.
        В 2D момент инерции инвариантен относительно поворота.
        Центр масс переводится напрямую.
        """
        c_new = pose.transform_point(self.c)
        return SpatialInertia2D(self.m, self.I_com, c_new)

    # ------------------------------
    #      Перенос инерции
    # ------------------------------
    def at_origin(self):
        """
        Перенос spatial inertia в фрейм, чей origin совпадает с self.c.
        То есть перенос COM → origin тела.
        """
        cx, cy = self.c
        m = self.m
        J = self.Jc

        # формула параллельного переноса
        J0 = J + m * (cx*cx + cy*cy)

        return SpatialInertia2D(m, J0, np.zeros(2))

    # ------------------------------
    #       Поворот инерции
    # ------------------------------
    def rotated(self, theta):
        """
        Повернуть spatial inertia 2D на угол theta.
        """
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
        ])

        # поворот центра масс
        c_new = R @ self.c

        # J переносится как скаляр (инвариант)
        return SpatialInertia2D(self.m, self.Jc, c_new)

    # ------------------------------
    #       Spatial inertia matrix
    # ------------------------------
    def to_matrix(self):
        m = self.m
        cx, cy = self.c
        J = self.Jc

        # spatial inertia в 2D (VW-порядок)
        return np.array([
            [m,     0,    -m*cy],
            [0,     m,     m*cx],
            [m*cy, -m*cx,  J + m*(cx*cx + cy*cy)]
        ], float)

    # ------------------------------
    #       Gravity wrench
    # ------------------------------
    def gravity_wrench(self, g):
        """
        Возвращает 3×1 винт (Fx, Fy, τz) в локальной системе!
        g — вектор гравитации в ЛОКАЛЬНОЙ системе.
        """
        m = self.m
        cx, cy = self.c

        F = m * g
        τ = cx * F[1] - cy * F[0]

        return np.array([F[0], F[1], τ], float)

    def __add__(self, other):
        if not isinstance(other, SpatialInertia2D):
            return NotImplemented

        m1, m2 = self.m, other.m
        c1, c2 = self.c, other.c
        J1, J2 = self.Jc, other.Jc

        m = m1 + m2
        if m == 0.0:
            # пустая инерция
            return SpatialInertia2D(0.0, 0.0, np.zeros(2))

        # общий центр масс
        c = (m1 * c1 + m2 * c2) / m

        # смещения от индивидуальных COM к общему
        d1 = c1 - c
        d2 = c2 - c

        # параллельный перенос для моментов инерции (вокруг общего COM)
        J = J1 + m1 * (d1 @ d1) + J2 + m2 * (d2 @ d2)

        return SpatialInertia2D(m, J, c)

    
    def get_kinetic_energy(self, velocity: np.ndarray, omega: float) -> float:
        v_squared = np.dot(velocity, velocity)
        return 0.5 * self.m * v_squared + 0.5 * self.I_com * omega**2