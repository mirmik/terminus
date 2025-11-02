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
                 omega_old: np.ndarray = None,         # скорость на предыдущем шаге
                 include_gyroscopic: bool = False):    # учитывать гироскопический момент
        """
        Args:
            omega: Переменная угловой скорости (размер 3)
            J: Тензор инерции (3×3 матрица) [кг·м²]
            B: Коэффициент демпфирования (скаляр или 3×3 матрица) [Н·м·с]
            dt: Шаг по времени [с]
            omega_old: Угловая скорость на предыдущем шаге [рад/с]
            include_gyroscopic: Учитывать гироскопический момент ω × (J*ω)
        """
        if omega.size != 3:
            raise ValueError("omega должна быть вектором размера 3")
        
        self.omega = omega
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
        
        self.dt = dt
        self.include_gyroscopic = include_gyroscopic
        
        if omega_old is None:
            self.omega_old = np.zeros(3)
        else:
            self.omega_old = np.asarray(omega_old)
            if self.omega_old.shape != (3,):
                raise ValueError("omega_old должен быть вектором размера 3")
        
        # Эффективная матрица для неявной схемы
        if dt is not None:
            if dt <= 0:
                raise ValueError("Шаг по времени должен быть положительным")
            # (J/dt + B)*ω_new = J/dt*ω_old + τ
            self.M_eff = self.J / dt + self.B
        else:
            # Статический анализ: просто демпфирование
            self.M_eff = self.B.copy()
            if np.allclose(self.M_eff, 0):
                self.M_eff = np.eye(3) * 1e-10  # малое число для стабильности
    
    def get_variables(self) -> List[Variable]:
        return [self.omega]
    
    def contribute_to_A(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Добавить эффективную матрицу в систему
        """
        indices = index_map[self.omega]
        for i in range(3):
            for j in range(3):
                A[indices[i], indices[j]] += self.M_eff[i, j]
    
    def contribute_to_b(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Добавить вклад от предыдущего состояния
        """
        if self.dt is None:
            return
        
        indices = index_map[self.omega]
        # Инерционный член: J/dt * ω_old
        inertia_term = (self.J / self.dt) @ self.omega_old
        
        for i in range(3):
            b[indices[i]] += inertia_term[i]
        
        # Гироскопический момент: -ω × (J*ω)
        # Примечание: это нелинейный член, здесь используется линеаризация
        if self.include_gyroscopic:
            gyro_term = -np.cross(self.omega_old, self.J @ self.omega_old)
            for i in range(3):
                b[indices[i]] += gyro_term[i]
    
    def update_state(self, omega_new: np.ndarray):
        """
        Обновить состояние после шага по времени
        """
        self.omega_old = np.asarray(omega_new).copy()
    
    def get_kinetic_energy(self, omega: np.ndarray = None) -> float:
        """
        Вычислить кинетическую энергию вращения
        
        Returns:
            E_rot = (1/2) * ω^T * J * ω [Дж]
        """
        if omega is None:
            omega = self.omega_old
        omega = np.asarray(omega)
        return 0.5 * omega @ self.J @ omega
    
    def get_angular_momentum(self, omega: np.ndarray = None) -> np.ndarray:
        """
        Вычислить угловой момент
        
        Returns:
            L = J * ω [кг·м²/с]
        """
        if omega is None:
            omega = self.omega_old
        return self.J @ omega


class TorqueVector3D(Contribution):
    """
    Векторный момент, приложенный к вращающемуся телу (3D).
    """
    
    def __init__(self, omega: Variable, torque: np.ndarray):
        """
        Args:
            omega: Переменная угловой скорости (размер 3)
            torque: Приложенный момент [τx, τy, τz] [Н·м]
        """
        if omega.size != 3:
            raise ValueError("omega должна быть вектором размера 3")
        
        self.omega = omega
        self.torque = np.asarray(torque)
        if self.torque.shape != (3,):
            raise ValueError("torque должен быть вектором размера 3")
    
    def get_variables(self) -> List[Variable]:
        return [self.omega]
    
    def contribute_to_A(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        pass
    
    def contribute_to_b(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
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
                 v_old: np.ndarray = None):    # скорость на предыдущем шаге
        """
        Args:
            velocity: Переменная скорости (размер 3)
            m: Масса [кг]
            C: Коэффициент демпфирования (скаляр или 3×3 матрица) [Н·с/м]
            dt: Шаг по времени [с]
            v_old: Скорость на предыдущем шаге [м/с]
        """
        if velocity.size != 3:
            raise ValueError("velocity должна быть вектором размера 3")
        
        if m <= 0:
            raise ValueError("Масса должна быть положительной")
        
        self.velocity = velocity
        self.m = m
        
        # Демпфирование
        if np.isscalar(C):
            self.C = np.eye(3) * C
        else:
            self.C = np.asarray(C)
            if self.C.shape != (3, 3):
                raise ValueError("C должен быть скаляром или матрицей 3×3")
        
        self.dt = dt
        
        if v_old is None:
            self.v_old = np.zeros(3)
        else:
            self.v_old = np.asarray(v_old)
            if self.v_old.shape != (3,):
                raise ValueError("v_old должен быть вектором размера 3")
        
        # Эффективная матрица
        if dt is not None:
            if dt <= 0:
                raise ValueError("Шаг по времени должен быть положительным")
            self.M_eff = (m / dt) * np.eye(3) + self.C
        else:
            self.M_eff = self.C.copy()
            if np.allclose(self.M_eff, 0):
                self.M_eff = np.eye(3) * 1e-10
    
    def get_variables(self) -> List[Variable]:
        return [self.velocity]
    
    def contribute_to_A(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        indices = index_map[self.velocity]
        for i in range(3):
            for j in range(3):
                A[indices[i], indices[j]] += self.M_eff[i, j]
    
    def contribute_to_b(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        if self.dt is None:
            return
        
        indices = index_map[self.velocity]
        inertia_term = (self.m / self.dt) * self.v_old
        
        for i in range(3):
            b[indices[i]] += inertia_term[i]
    
    def update_state(self, v_new: np.ndarray):
        """
        Обновить состояние после шага по времени
        """
        self.v_old = np.asarray(v_new).copy()
    
    def get_kinetic_energy(self, v: np.ndarray = None) -> float:
        """
        Вычислить кинетическую энергию
        
        Returns:
            E = (1/2) * m * v² [Дж]
        """
        if v is None:
            v = self.v_old
        v = np.asarray(v)
        return 0.5 * self.m * np.dot(v, v)
    
    def get_momentum(self, v: np.ndarray = None) -> np.ndarray:
        """
        Вычислить линейный импульс
        
        Returns:
            p = m * v [кг·м/с]
        """
        if v is None:
            v = self.v_old
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
                 v_old: np.ndarray = None,             # [vx, vy, vz] старая
                 omega_old: np.ndarray = None,         # [ωx, ωy, ωz] старая
                 include_gyroscopic: bool = False):    # учитывать гироскопический момент
        """
        Args:
            velocity: Переменная линейной скорости (размер 3)
            omega: Переменная угловой скорости (размер 3)
            m: Масса [кг]
            J: Тензор инерции (3×3 матрица) [кг·м²]
            C: Коэффициент линейного демпфирования [Н·с/м]
            B: Коэффициент углового демпфирования [Н·м·с]
            dt: Шаг по времени [с]
            v_old: Скорость на предыдущем шаге
            omega_old: Угловая скорость на предыдущем шаге
            include_gyroscopic: Учитывать гироскопический момент
        """
        if velocity.size != 3:
            raise ValueError("velocity должна быть вектором размера 3")
        
        if omega.size != 3:
            raise ValueError("omega должна быть вектором размера 3")
        
        if m <= 0:
            raise ValueError("Масса должна быть положительной")
        
        self.velocity = velocity
        self.omega = omega
        self.m = m
        
        # Создаем внутренние элементы
        self.mass = LinearMass3D(velocity, m, C, dt, v_old)
        self.inertia = RotationalInertia3D(omega, J, B, dt, omega_old, include_gyroscopic)
    
    def get_variables(self) -> List[Variable]:
        return [self.velocity, self.omega]
    
    def contribute_to_A(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        self.mass.contribute_to_A(A, index_map)
        self.inertia.contribute_to_A(A, index_map)
    
    def contribute_to_b(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        self.mass.contribute_to_b(b, index_map)
        self.inertia.contribute_to_b(b, index_map)
    
    def update_state(self, v_new: np.ndarray, omega_new: np.ndarray):
        """
        Обновить состояние после шага по времени
        """
        self.mass.update_state(v_new)
        self.inertia.update_state(omega_new)
    
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
    
    def __init__(self, velocity: Variable, force: np.ndarray):
        """
        Args:
            velocity: Переменная скорости (размер 3)
            force: Приложенная сила [Fx, Fy, Fz] [Н]
        """
        if velocity.size != 3:
            raise ValueError("velocity должна быть вектором размера 3")
        
        self.velocity = velocity
        self.force = np.asarray(force)
        if self.force.shape != (3,):
            raise ValueError("force должна быть вектором размера 3")
    
    def get_variables(self) -> List[Variable]:
        return [self.velocity]
    
    def contribute_to_A(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        pass
    
    def contribute_to_b(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
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
                 r: np.ndarray):
        """
        Args:
            velocity: Переменная скорости центра масс (размер 3)
            omega: Переменная угловой скорости (размер 3)
            r: Вектор от ЦМ к точке шарнира [rx, ry, rz] [м]
        """
        if velocity.size != 3:
            raise ValueError("velocity должна быть вектором размера 3")
        
        if omega.size != 3:
            raise ValueError("omega должна быть вектором размера 3")
        
        self.velocity = velocity
        self.omega = omega
        self.r = np.asarray(r)
        
        if self.r.shape != (3,):
            raise ValueError("r должен быть вектором размера 3")
        
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
    
    def get_variables(self) -> List[Variable]:
        return [self.velocity, self.omega]
    
    def get_n_constraints(self) -> int:
        """Три уравнения связи (vx, vy, vz)"""
        return 3
    
    def contribute_to_C(self, C: np.ndarray, constraint_offset: int,
                       index_map: Dict[Variable, List[int]]):
        """
        Связь: v_cm + ω × r = 0
        v + [r]_× * ω = 0
        """
        v_indices = index_map[self.velocity]
        omega_indices = index_map[self.omega]
        
        # Коэффициенты при v: единичная матрица
        for i in range(3):
            C[constraint_offset + i, v_indices[i]] += 1.0
        
        # Коэффициенты при ω: кососимметрическая матрица [r]_×
        for i in range(3):
            for j in range(3):
                C[constraint_offset + i, omega_indices[j]] += self.r_skew[i, j]
    
    def contribute_to_d(self, d: np.ndarray, constraint_offset: int):
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
    
    def __init__(self, velocity: Variable, target: np.ndarray = None):
        """
        Args:
            velocity: Переменная скорости (размер 3)
            target: Целевая скорость [vx, vy, vz] (по умолчанию [0, 0, 0])
        """
        if velocity.size != 3:
            raise ValueError("velocity должна быть вектором размера 3")
        
        self.velocity = velocity
        
        if target is None:
            self.target = np.zeros(3)
        else:
            self.target = np.asarray(target)
            if self.target.shape != (3,):
                raise ValueError("target должен быть вектором размера 3")
    
    def get_variables(self) -> List[Variable]:
        return [self.velocity]
    
    def get_n_constraints(self) -> int:
        """Три уравнения связи"""
        return 3
    
    def contribute_to_C(self, C: np.ndarray, constraint_offset: int,
                       index_map: Dict[Variable, List[int]]):
        """
        Матрица связи: единичная матрица
        """
        v_indices = index_map[self.velocity]
        
        for i in range(3):
            C[constraint_offset + i, v_indices[i]] += 1.0
    
    def contribute_to_d(self, d: np.ndarray, constraint_offset: int):
        """
        Правая часть: целевая скорость
        """
        d[constraint_offset:constraint_offset + 3] += self.target


class FixedRotation3D(Constraint):
    """
    Фиксация вращения (ω = target).
    """
    
    def __init__(self, omega: Variable, target: np.ndarray = None):
        """
        Args:
            omega: Переменная угловой скорости (размер 3)
            target: Целевая угловая скорость [ωx, ωy, ωz] (по умолчанию [0, 0, 0])
        """
        if omega.size != 3:
            raise ValueError("omega должна быть вектором размера 3")
        
        self.omega = omega
        
        if target is None:
            self.target = np.zeros(3)
        else:
            self.target = np.asarray(target)
            if self.target.shape != (3,):
                raise ValueError("target должен быть вектором размера 3")
    
    def get_variables(self) -> List[Variable]:
        return [self.omega]
    
    def get_n_constraints(self) -> int:
        """Три уравнения связи"""
        return 3
    
    def contribute_to_C(self, C: np.ndarray, constraint_offset: int,
                       index_map: Dict[Variable, List[int]]):
        """
        Матрица связи: единичная матрица
        """
        omega_indices = index_map[self.omega]
        
        for i in range(3):
            C[constraint_offset + i, omega_indices[i]] += 1.0
    
    def contribute_to_d(self, d: np.ndarray, constraint_offset: int):
        """
        Правая часть: целевая угловая скорость
        """
        d[constraint_offset:constraint_offset + 3] += self.target
