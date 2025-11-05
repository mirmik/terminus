#!/usr/bin/env python3
"""
Элементы многотельной механики для пространственного (3D) движения.

Реализованы элементы для моделирования пространственных механических систем:

Contributions (физические элементы):
- RotationalInertia3D: вращательная инерция с тензором инерции (3 DOF: ωx, ωy, ωz)
- TorqueVector3D: векторный момент на тело
- LinearMass3D: точечная масса с демпфированием (3 DOF: vx, vy, vz)
- RigidBody3D: твердое тело (6 DOF: vx, vy, vz, ωx, ωy, ωz)
- ForceVector3D: векторная сила на тело

Constraints (кинематические связи):
- SphericalJoint3D: сферический шарнир (точка фиксирована)
- FixedPoint3D: фиксация точки в пространстве
- FixedRotation3D: фиксация вращения
"""

import numpy as np
from typing import List, Dict, Union
from .assembler import Contribution, Constraint, Variable


# ============================================================================
# Contributions (физические элементы)
# ============================================================================


class RotationalInertia3D(Contribution):
    """
    Вращательная инерция для пространственного движения.
    
    Уравнение Эйлера для твердого тела:
    J*dω/dt + ω × (J*ω) = Σ τ - B*ω
    
    где:
    - J: тензор инерции [кг·м²] (3×3 матрица)
    - ω: вектор угловой скорости [ωx, ωy, ωz] [рад/с]
    - B: коэффициент вязкого трения [Н·м·с] (скаляр или 3×3 матрица)
    - τ: внешние моменты [Н·м]
    
    Примечание: гироскопический член ω × (J*ω) нелинейный и требует
    линеаризации или итеративного решения.
    """
    
    def __init__(self,
                 omega: Variable,                      # угловая скорость [ωx, ωy, ωz]
                 J: np.ndarray,                        # тензор инерции [кг·м²]
                 B: Union[float, np.ndarray] = 0.0,    # демпфирование [Н·м·с]
                 dt: float = None,                     # шаг по времени [с]
                 include_gyroscopic: bool = False,     # учитывать гироскопический момент
                 assembler=None):                      # ассемблер для автоматической регистрации
        """
        Args:
            omega: Переменная угловой скорости (размер 3)
            J: Тензор инерции (3×3 матрица) [кг·м²]
            B: Коэффициент демпфирования (скаляр или 3×3 матрица) [Н·м·с]
            dt: Шаг по времени [с]
            include_gyroscopic: Учитывать гироскопический момент ω × (J*ω)
            assembler: MatrixAssembler для автоматической регистрации переменных
        """
        if omega.size != 3:
            raise ValueError("omega должна быть вектором размера 3")
        
        self.J = np.asarray(J)
        if self.J.shape != (3, 3):
            raise ValueError("J должен быть матрицей 3×3")
        
        # Проверка симметричности тензора инерции
        if not np.allclose(self.J, self.J.T):
            raise ValueError("Тензор инерции J должен быть симметричным")
        
        # Демпфирование
        if np.isscalar(B):
            self.B = np.eye(3) * B
        else:
            self.B = np.asarray(B)
            if self.B.shape != (3, 3):
                raise ValueError("B должен быть скаляром или матрицей 3×3")
        
        if dt is not None and dt <= 0:
            raise ValueError("Шаг по времени должен быть положительным")
        
        super().__init__([omega], assembler)
        
        self.omega = omega
        self.dt = dt
        self.include_gyroscopic = include_gyroscopic
        
        # Эффективная матрица для неявной схемы
        if dt is not None:
            # (J/dt + B)*ω_new = J/dt*ω_old + τ
            self.M_eff = self.J / dt + self.B
        else:
            # Статический анализ: просто демпфирование
            self.M_eff = self.B.copy()
            if np.allclose(self.M_eff, 0):
                self.M_eff = np.eye(3) * 1e-10  # малое число для стабильности
    
    def contribute_to_mass(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Добавить эффективную матрицу в систему
        """
        indices = index_map[self.omega]
        for i in range(3):
            for j in range(3):
                A[indices[i], indices[j]] += self.M_eff[i, j]
    
    def contribute_to_load(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Добавить вклад от предыдущего состояния
        """
        if self.dt is None:
            return
        
        indices = index_map[self.omega]
        # Инерционный член: J/dt * ω_old
        inertia_term = (self.J / self.dt) @ self.omega.value
        
        for i in range(3):
            b[indices[i]] += inertia_term[i]
        
        # Гироскопический момент: -ω × (J*ω)
        # Примечание: это нелинейный член, здесь используется линеаризация
        if self.include_gyroscopic:
            gyro_term = -np.cross(self.omega.value, self.J @ self.omega.value)
            for i in range(3):
                b[indices[i]] += gyro_term[i]
    
    def get_kinetic_energy(self, omega: np.ndarray = None) -> float:
        """
        Вычислить кинетическую энергию вращения
        
        Returns:
            E_rot = (1/2) * ω^T * J * ω [Дж]
        """
        if omega is None:
            omega = self.omega.value
        omega = np.asarray(omega)
        return 0.5 * omega @ self.J @ omega
    
    def get_angular_momentum(self, omega: np.ndarray = None) -> np.ndarray:
        """
        Вычислить угловой момент
        
        Returns:
            L = J * ω [кг·м²/с]
        """
        if omega is None:
            omega = self.omega.value
        return self.J @ omega


class TorqueVector3D(Contribution):
    """
    Векторный момент, приложенный к вращающемуся телу (3D).
    """
    
    def __init__(self, omega: Variable, torque: np.ndarray, assembler=None):
        """
        Args:
            omega: Переменная угловой скорости (размер 3)
            torque: Приложенный момент [τx, τy, τz] [Н·м]
            assembler: MatrixAssembler для автоматической регистрации переменных
        """
        if omega.size != 3:
            raise ValueError("omega должна быть вектором размера 3")
        
        self.torque = np.asarray(torque)
        if self.torque.shape != (3,):
            raise ValueError("torque должен быть вектором размера 3")
        
        super().__init__([omega], assembler)
        
        self.omega = omega
    
    def contribute_to_mass(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        pass
    
    def contribute_to_load(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        indices = index_map[self.omega]
        for i in range(3):
            b[indices[i]] += self.torque[i]
    
    def get_power(self, omega: np.ndarray) -> float:
        """
        Вычислить мощность источника момента
        
        Returns:
            P = τ · ω [Вт]
        """
        return np.dot(self.torque, omega)


class LinearMass3D(Contribution):
    """
    Точечная масса с линейным демпфированием (3D).
    
    Уравнение: m*dv/dt = Σ F - C*v
    
    где:
    - m: масса [кг]
    - v: линейная скорость [vx, vy, vz] [м/с]
    - C: коэффициент вязкого сопротивления [Н·с/м]
    - F: внешние силы [Н]
    """
    
    def __init__(self,
                 velocity: Variable,           # линейная скорость
                 m: float,                     # масса [кг]
                 C: Union[float, np.ndarray] = 0.0,  # демпфирование [Н·с/м]
                 dt: float = None,             # шаг по времени [с]
                 assembler=None):              # ассемблер для автоматической регистрации
        """
        Args:
            velocity: Переменная скорости (размер 3)
            m: Масса [кг]
            C: Коэффициент демпфирования (скаляр или 3×3 матрица) [Н·с/м]
            dt: Шаг по времени [с]
            assembler: MatrixAssembler для автоматической регистрации переменных
        """
        if velocity.size != 3:
            raise ValueError("velocity должна быть вектором размера 3")
        
        if m <= 0:
            raise ValueError("Масса должна быть положительной")
        
        # Демпфирование
        if np.isscalar(C):
            self.C = np.eye(3) * C
        else:
            self.C = np.asarray(C)
            if self.C.shape != (3, 3):
                raise ValueError("C должен быть скаляром или матрицей 3×3")
        
        if dt is not None and dt <= 0:
            raise ValueError("Шаг по времени должен быть положительным")
        
        super().__init__([velocity], assembler)
        
        self.velocity = velocity
        self.m = m
        self.dt = dt
        
        # Эффективная матрица
        if dt is not None:
            self.M_eff = (m / dt) * np.eye(3) + self.C
        else:
            self.M_eff = self.C.copy()
            if np.allclose(self.M_eff, 0):
                self.M_eff = np.eye(3) * 1e-10
    
    def contribute_to_mass(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        indices = index_map[self.velocity]
        for i in range(3):
            for j in range(3):
                A[indices[i], indices[j]] += self.M_eff[i, j]
    
    def contribute_to_load(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        if self.dt is None:
            return
        
        indices = index_map[self.velocity]
        inertia_term = (self.m / self.dt) * self.velocity.value
        
        for i in range(3):
            b[indices[i]] += inertia_term[i]
    
    def get_kinetic_energy(self, v: np.ndarray = None) -> float:
        """
        Вычислить кинетическую энергию
        
        Returns:
            E = (1/2) * m * v² [Дж]
        """
        if v is None:
            v = self.velocity.value
        v = np.asarray(v)
        return 0.5 * self.m * np.dot(v, v)
    
    def get_momentum(self, v: np.ndarray = None) -> np.ndarray:
        """
        Вычислить линейный импульс
        
        Returns:
            p = m * v [кг·м/с]
        """
        if v is None:
            v = self.velocity.value
        return self.m * np.asarray(v)


class RigidBody3D(Contribution):
    """
    Твердое тело в пространстве (6 степеней свободы: vx, vy, vz, ωx, ωy, ωz).
    
    Объединяет поступательное и вращательное движение:
    - Поступательное: m*dv/dt = F - C*v
    - Вращательное: J*dω/dt + ω × (J*ω) = τ - B*ω
    """
    
    def __init__(self,
                 velocity: Variable,                   # линейная скорость [vx, vy, vz]
                 omega: Variable,                      # угловая скорость [ωx, ωy, ωz]
                 m: float,                             # масса [кг]
                 J: np.ndarray,                        # тензор инерции [кг·м²]
                 C: Union[float, np.ndarray] = 0.0,    # линейное демпфирование [Н·с/м]
                 B: Union[float, np.ndarray] = 0.0,    # угловое демпфирование [Н·м·с]
                 dt: float = None,                     # шаг по времени [с]
                 include_gyroscopic: bool = False,     # учитывать гироскопический момент
                 assembler=None):                      # ассемблер для автоматической регистрации
        """
        Args:
            velocity: Переменная линейной скорости (размер 3)
            omega: Переменная угловой скорости (размер 3)
            m: Масса [кг]
            J: Тензор инерции (3×3 матрица) [кг·м²]
            C: Коэффициент линейного демпфирования [Н·с/м]
            B: Коэффициент углового демпфирования [Н·м·с]
            dt: Шаг по времени [с]
            include_gyroscopic: Учитывать гироскопический момент
            assembler: MatrixAssembler для автоматической регистрации переменных
        """
        if velocity.size != 3:
            raise ValueError("velocity должна быть вектором размера 3")
        
        if omega.size != 3:
            raise ValueError("omega должна быть вектором размера 3")
        
        if m <= 0:
            raise ValueError("Масса должна быть положительной")
        
        super().__init__([velocity, omega], assembler)
        
        self.velocity = velocity
        self.omega = omega
        self.m = m
        
        # Создаем внутренние элементы
        self.mass = LinearMass3D(velocity, m, C, dt)
        self.inertia = RotationalInertia3D(omega, J, B, dt, include_gyroscopic)
    
    def contribute_to_mass(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        self.mass.contribute_to_mass(A, index_map)
        self.inertia.contribute_to_mass(A, index_map)
    
    def contribute_to_load(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        self.mass.contribute_to_load(b, index_map)
        self.inertia.contribute_to_load(b, index_map)
    
    def get_kinetic_energy(self, v: np.ndarray = None, omega: np.ndarray = None) -> float:
        """
        Вычислить полную кинетическую энергию
        
        Returns:
            E = (1/2)*m*v² + (1/2)*ω^T*J*ω [Дж]
        """
        E_trans = self.mass.get_kinetic_energy(v)
        E_rot = self.inertia.get_kinetic_energy(omega)
        return E_trans + E_rot
    
    def get_linear_momentum(self, v: np.ndarray = None) -> np.ndarray:
        """
        Вычислить линейный импульс
        """
        return self.mass.get_momentum(v)
    
    def get_angular_momentum(self, omega: np.ndarray = None) -> np.ndarray:
        """
        Вычислить угловой момент
        """
        return self.inertia.get_angular_momentum(omega)


class ForceVector3D(Contribution):
    """
    Векторная сила, приложенная к телу (3D).
    """
    
    def __init__(self, velocity: Variable, force: np.ndarray, assembler=None):
        """
        Args:
            velocity: Переменная скорости (размер 3)
            force: Приложенная сила [Fx, Fy, Fz] [Н]
            assembler: MatrixAssembler для автоматической регистрации переменных
        """
        if velocity.size != 3:
            raise ValueError("velocity должна быть вектором размера 3")
        
        self.force = np.asarray(force)
        if self.force.shape != (3,):
            raise ValueError("force должна быть вектором размера 3")
        
        super().__init__([velocity], assembler)
        
        self.velocity = velocity
    
    def contribute_to_mass(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        pass
    
    def contribute_to_load(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        indices = index_map[self.velocity]
        for i in range(3):
            b[indices[i]] += self.force[i]
    
    def get_power(self, v: np.ndarray) -> float:
        """
        Вычислить мощность источника силы
        
        Returns:
            P = F · v [Вт]
        """
        return np.dot(self.force, v)


# ============================================================================
# Constraints (кинематические связи)
# ============================================================================


class SphericalJoint3D(Constraint):
    """
    Сферический шарнир (spherical joint, ball joint) для твердого тела.
    
    Фиксирует точку на теле в пространстве, разрешая только вращение вокруг этой точки.
    
    Кинематическая связь:
    v_cm + ω × r = 0
    
    где:
    - v_cm: скорость центра масс [vx, vy, vz]
    - ω: угловая скорость [ωx, ωy, ωz]
    - r: вектор от центра масс к точке крепления [rx, ry, rz]
    """
    
    def __init__(self,
                 velocity: Variable,
                 omega: Variable,
                 r: np.ndarray,
                 assembler=None):
        """
        Args:
            velocity: Переменная скорости центра масс (размер 3)
            omega: Переменная угловой скорости (размер 3)
            r: Вектор от ЦМ к точке шарнира [rx, ry, rz] [м]
            assembler: MatrixAssembler для автоматической регистрации переменных
        """
        if velocity.size != 3:
            raise ValueError("velocity должна быть вектором размера 3")
        
        if omega.size != 3:
            raise ValueError("omega должна быть вектором размера 3")
        
        self.r = np.asarray(r)
        
        if self.r.shape != (3,):
            raise ValueError("r должен быть вектором размера 3")

        self.lambdas = Variable(size=3, name="lambda_spherical_joint_3d")
        
        super().__init__([velocity, omega], [self.lambdas], [], assembler)
        
        self.velocity = velocity
        self.omega = omega
        
        # Матрица кососимметрическая для векторного произведения: ω × r
        # ω × r = -r × ω = -[r]_× * ω
        # где [r]_× - стандартная кососимметрическая матрица для r
        # [r]_× = [  0   -rz   ry ]
        #         [ rz    0   -rx ]
        #         [-ry   rx    0  ]
        # Поэтому для v + ω × r = 0 получаем: v - [r]_× * ω = 0
        r_skew_standard = np.array([
            [    0,  -r[2],   r[1]],
            [ r[2],      0,  -r[0]],
            [-r[1],   r[0],      0]
        ])
        # Берем с обратным знаком: ω × r = -[r]_× * ω
        self.r_skew = -r_skew_standard
    
    
    def contribute_to_holonomic(self, C: np.ndarray, 
                       index_map: Dict[Variable, List[int]],
                       lambdas_index_map: Dict[Variable, List[int]]):
        """
        Связь: v_cm + ω × r = 0
        v + [r]_× * ω = 0
        """
        v_indices = index_map[self.velocity]
        omega_indices = index_map[self.omega]

        constr_indices = lambdas_index_map[self.lambdas]
        
        # Коэффициенты при v: единичная матрица
        for i in range(3):
            C[constr_indices[i], v_indices[i]] += 1.0
        
        # Коэффициенты при ω: кососимметрическая матрица [r]_×
        for i in range(3):
            for j in range(3):
                C[constr_indices[i], omega_indices[j]] += self.r_skew[i, j]

    def contribute_to_holonomic_load(self, d: np.ndarray, lambdas_index_map: Dict[Variable, List[int]]):
        """
        Правая часть связи (нулевая)
        """
        pass
    
    def get_constraint_violation(self, v: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """
        Вычислить нарушение кинематической связи
        
        Returns:
            Вектор нарушения: v + ω × r [м/с]
        """
        return v + np.cross(omega, self.r)


class FixedPoint3D(Constraint):
    """
    Фиксация точки в пространстве (v = target).
    """
    
    def __init__(self, velocity: Variable, target: np.ndarray = None, assembler=None):
        """
        Args:
            velocity: Переменная скорости (размер 3)
            target: Целевая скорость [vx, vy, vz] (по умолчанию [0, 0, 0])
            assembler: MatrixAssembler для автоматической регистрации переменных
        """
        if velocity.size != 3:
            raise ValueError("velocity должна быть вектором размера 3")
        
        if target is None:
            self.target = np.zeros(3)
        else:
            self.target = np.asarray(target)
            if self.target.shape != (3,):
                raise ValueError("target должен быть вектором размера 3")   

        self.lambdas = Variable(size=3, name="lambda_fixed_point_3d")

        super().__init__([velocity], [self.lambdas], [], assembler)

        self.velocity = velocity


    def contribute_to_holonomic(self, C: np.ndarray,
                       index_map: Dict[Variable, List[int]],
                       lambdas_index_map: Dict[Variable, List[int]],):
        """
        Матрица связи: единичная матрица
        """
        v_indices = index_map[self.velocity]
        lambdas_indices = lambdas_index_map[self.lambdas]
        
        for i in range(3):
            C[lambdas_indices[i], v_indices[i]] += 1.0

    def contribute_to_holonomic_load(self, d: np.ndarray, lambdas_index_map: Dict[Variable, List[int]]):
        """
        Правая часть: целевая скорость
        """
        d[lambdas_index_map[self.lambdas]] += self.target


class FixedRotation3D(Constraint):
    """
    Фиксация вращения (ω = target).
    """
    
    def __init__(self, omega: Variable, target: np.ndarray = None, assembler=None):
        """
        Args:
            omega: Переменная угловой скорости (размер 3)
            target: Целевая угловая скорость [ωx, ωy, ωz] (по умолчанию [0, 0, 0])
            assembler: MatrixAssembler для автоматической регистрации переменных
        """
        if omega.size != 3:
            raise ValueError("omega должна быть вектором размера 3")
        
        if target is None:
            self.target = np.zeros(3)
        else:
            self.target = np.asarray(target)
            if self.target.shape != (3,):
                raise ValueError("target должен быть вектором размера 3")

        self.lambdas = Variable(size=3, name="lambda_fixed_rotation_3d")

        super().__init__([omega], [self.lambdas], [], assembler)

        self.omega = omega


    def contribute_to_holonomic(self, C: np.ndarray,
                       index_map: Dict[Variable, List[int]],
                       lambdas_index_map: Dict[Variable, List[int]]):
        """
        Матрица связи: единичная матрица
        """
        omega_indices = index_map[self.omega]
        lambdas_indices = lambdas_index_map[self.lambdas]
        
        for i in range(3):
            C[lambdas_indices[i], omega_indices[i]] += 1.0

    def contribute_to_holonomic_load(self, d: np.ndarray, lambdas_index_map: Dict[Variable, List[int]]):
        """
        Правая часть: целевая угловая скорость
        """
        d[lambdas_index_map[self.lambdas]] += self.target
