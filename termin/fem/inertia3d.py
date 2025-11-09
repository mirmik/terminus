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
    def __init__(self, mass, inertia, com=np.zeros(3)):
        self.m = float(mass)
        self.c = np.asarray(com).reshape(3)     # COM
        self.I_com = np.asarray(inertia).reshape(3,3)

    def transform_by(self, pose: Pose3):
        R = pose.rotation_matrix()
        c_new = pose.transform_point(self.c)   # перевод COM в мировой фрейм (с трансляцией)
        I_rot = R @ self.I_com @ R.T           # всё ещё центральный тензор, только в мировом базисе
        return SpatialInertia3D(self.m, I_rot, c_new)

    def rotate_by(self, pose: Pose3) -> "SpatialInertia3D":
        """
        Повернуть spatial inertia согласно ориентации pose.
        Трансляция pose игнорируется — spatial inertia зависит только от ориентации.
        COM и inertia переносятся поворотом.
        """

        R = pose.rotation_matrix()

        # 1) Поворот центра масс
        c_new = R @ self.c

        # 2) Поворот центрального тензора инерции
        I_com_new = R @ self.I_com @ R.T

        return SpatialInertia3D(self.m, I_com_new, c_new)

    def at_body_origin(self) -> "SpatialInertia3D":
        """
        Возвращает spatial inertia, приведённую к origin тела.
        COM становится (0,0,0).
        Тензор инерции пересчитывается по формуле Штейнера.
        """

        c = self.c              # COM в локальной системе тела
        m = self.m
        S = skew(c)

        # инерция относительно origin тела (а не COM)
        I_origin = self.I_com + m * (S @ S.T)

        # после переноса COM должен быть равен нулю
        return SpatialInertia3D(m, I_origin, com=np.zeros(3))

    def at_body_origin_wv_order(self):
        c = self.c
        m = self.m
        S = skew(c)

        I_com = self.I_com
        I_origin = I_com + m * (S @ S.T)

        upper_left  = I_origin
        upper_right = m * S
        lower_left  = -m * S.T
        lower_right = m * np.eye(3)

        return np.block([
            [upper_left,  upper_right],
            [lower_left,  lower_right]
        ])

    def at_body_origin_vw_order(self):
        c = self.c
        m = self.m
        S = skew(c)

        I_com = self.I_com
        I_origin = I_com + m * (S @ S.T)

        # WV-order blocks
        A = I_origin
        B = m * S
        C = -m * S.T
        D = m * np.eye(3)

        # VW permutation
        return np.block([
            [D,  C],
            [B,  A]
        ])


    def gravity_wrench(self, gravity_local: np.ndarray) -> Screw3:
        """
        Возвращает пространственный винт (сила+момент) от гравитации,
        выраженный в ЛОКАЛЬНОЙ системе тела.

        gravity_local — гравитация, выраженная в локальной системе.
        """

        F = self.m * gravity_local
        τ = np.cross(self.c, F)  # self.c — локальный COM

        return Screw3(ang=τ, lin=F)

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

    def to_matrix_wv_order(self):
        """
        Spatial inertia matrix в порядке [ω; v].
        """
        csk = skew(self.c)
        m = self.m
        I = self.I_com

        upper_left  = I + m * (csk @ csk.T)
        upper_right = m * csk
        lower_left  = m * csk.T
        lower_right = m * np.eye(3)

        return np.block([
            [upper_left,  upper_right],
            [lower_left,  lower_right]
        ])


    def to_matrix_vw_order(self):
        """
        Spatial inertia matrix в порядке [v; ω].
        Это просто перестановка блоков wv-матрицы.
        """
        csk = skew(self.c)
        m = self.m
        I = self.I_com

        # блоки те же самые
        upper_left  = m * np.eye(3)            # v–v
        upper_right = m * csk.T               # v–ω
        lower_left  = m * csk                  # ω–v
        lower_right = I + m * (csk @ csk.T)    # ω–ω

        return np.block([
            [upper_left,  upper_right],
            [lower_left,  lower_right]
        ])