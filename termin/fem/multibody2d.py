#!/usr/bin/env python3
"""
Элементы многотельной механики для плоского (2D) движения.

Реализованы элементы для моделирования планарных механических систем:

Contributions (физические элементы):
- RotationalInertia2D: вращательная инерция с демпфированием
- TorqueSource2D: источник момента
- RigidBody2D: твердое тело (3 DOF: vx, vy, ω)
- ForceVector2D: векторная сила на тело

Constraints (кинематические связи):
- RevoluteJoint2D: вращательный шарнир (точка фиксирована)
- FixedPoint2D: фиксация точки в пространстве
"""

import numpy as np
from typing import List, Dict
from .assembler import Contribution, Constraint, Variable


# ============================================================================
# Contributions (физические элементы)
# ============================================================================


class RotationalInertia2D(Contribution):
    """
    Вращательная инерция для плоского движения (вращение вокруг одной оси).
    
    Уравнение: J*dω/dt = Σ τ - B*ω
    
    где:
    - J: момент инерции [кг·м²]
    - ω: угловая скорость [рад/с]
    - B: коэффициент вязкого трения [Н·м·с]
    - τ: внешние моменты [Н·м]
    
    Для интегрирования используется неявная схема Эйлера.
    """
    
    def __init__(self,
                 omega: Variable,         # угловая скорость
                 J: float,                # момент инерции [кг·м²]
                 B: float = 0.0,          # коэффициент демпфирования [Н·м·с]
                 dt: float = None,        # шаг по времени [с]
                 omega_old: float = 0.0): # скорость на предыдущем шаге
        """
        Args:
            omega: Переменная угловой скорости
            J: Момент инерции [кг·м²]
            B: Коэффициент вязкого трения [Н·м·с]
            dt: Шаг по времени [с]
            omega_old: Угловая скорость на предыдущем шаге [рад/с]
        """
        if omega.size != 1:
            raise ValueError("omega должна быть скаляром")
        
        if J <= 0:
            raise ValueError("Момент инерции должен быть положительным")
        
        if B < 0:
            raise ValueError("Коэффициент демпфирования не может быть отрицательным")
        
        self.omega = omega
        self.J = J
        self.B = B
        self.dt = dt
        self.omega_old = omega_old
        
        if dt is not None:
            if dt <= 0:
                raise ValueError("Шаг по времени должен быть положительным")
            # Эффективный коэффициент для неявной схемы
            # J*(ω_new - ω_old)/dt + B*ω_new = τ_external
            # (J/dt + B)*ω_new = τ_external + J/dt*ω_old
            self.C_eff = J / dt + B
        else:
            # Статический анализ: просто демпфирование
            self.C_eff = B if B > 0 else 1e-10  # малое число для численной стабильности
    
    def get_variables(self) -> List[Variable]:
        return [self.omega]
    
    def contribute_to_A(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Добавить эффективный коэффициент в диагональ
        """
        idx = index_map[self.omega][0]
        A[idx, idx] += self.C_eff
    
    def contribute_to_b(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Добавить вклад от предыдущего состояния
        """
        if self.dt is None:
            return
        
        idx = index_map[self.omega][0]
        # Инерционный член от предыдущего шага
        b[idx] += (self.J / self.dt) * self.omega_old
    
    def update_state(self, omega_new: float):
        """
        Обновить состояние после шага по времени
        """
        self.omega_old = omega_new
    
    def get_kinetic_energy(self, omega: float = None) -> float:
        """
        Вычислить кинетическую энергию вращения
        
        Args:
            omega: Угловая скорость [рад/с] (если None, используется текущая)
        
        Returns:
            Кинетическая энергия E_k = (1/2)*J*ω² [Дж]
        """
        if omega is None:
            omega = self.omega_old
        return 0.5 * self.J * omega**2


class TorqueSource2D(Contribution):
    """
    Источник момента (внешний момент, приложенный к вращающемуся телу).
    
    Аналог источника тока для механической системы.
    """
    
    def __init__(self, omega: Variable, torque: float):
        """
        Args:
            omega: Переменная угловой скорости
            torque: Приложенный момент [Н·м] (положительный = ускорение)
        """
        if omega.size != 1:
            raise ValueError("omega должна быть скаляром")
        
        self.omega = omega
        self.torque = torque
    
    def get_variables(self) -> List[Variable]:
        return [self.omega]
    
    def contribute_to_A(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Источник момента не влияет на матрицу
        """
        pass
    
    def contribute_to_b(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Добавить момент в правую часть
        """
        idx = index_map[self.omega][0]
        b[idx] += self.torque
    
    def get_power(self, omega: float) -> float:
        """
        Вычислить мощность источника момента
        
        Args:
            omega: Угловая скорость [рад/с]
        
        Returns:
            Мощность P = τ*ω [Вт]
        """
        return self.torque * omega


class RigidBody2D(Contribution):
    """
    Твердое тело в плоскости (3 степени свободы: x, y, θ).
    
    Объединяет поступательное и вращательное движение:
    - Поступательное: m*dv/dt = F (вектор в плоскости XY)
    - Вращательное: J*dω/dt = τ (вращение вокруг оси Z)
    
    Переменные:
    - velocity: Variable размера 2 для [vx, vy]
    - omega: Variable размера 1 для угловой скорости ω
    
    Уравнения движения (неявная схема Эйлера):
    (m/dt)*v_new + C*v_new = (m/dt)*v_old + F
    (J/dt)*ω_new + B*ω_new = (J/dt)*ω_old + τ
    """
    
    def __init__(self,
                 velocity: Variable,      # линейная скорость [vx, vy] (размер 2)
                 omega: Variable,         # угловая скорость ω (размер 1)
                 m: float,                # масса [кг]
                 J: float,                # момент инерции [кг·м²]
                 C: float = 0.0,          # коэффициент линейного демпфирования [Н·с/м]
                 B: float = 0.0,          # коэффициент углового демпфирования [Н·м·с]
                 dt: float = None,        # шаг по времени [с]
                 v_old: np.ndarray = None,  # [vx_old, vy_old]
                 omega_old: float = 0.0):   # ω_old
        """
        Args:
            velocity: Переменная линейной скорости (размер 2)
            omega: Переменная угловой скорости (размер 1)
            m: Масса [кг]
            J: Момент инерции [кг·м²]
            C: Коэффициент вязкого сопротивления для поступательного движения [Н·с/м]
            B: Коэффициент вязкого трения для вращательного движения [Н·м·с]
            dt: Шаг по времени [с]
            v_old: Скорость на предыдущем шаге [vx, vy]
            omega_old: Угловая скорость на предыдущем шаге [рад/с]
        """
        if velocity.size != 2:
            raise ValueError("velocity должна иметь размер 2 (vx, vy)")
        
        if omega.size != 1:
            raise ValueError("omega должна быть скаляром")
        
        if m <= 0:
            raise ValueError("Масса должна быть положительной")
        
        if J <= 0:
            raise ValueError("Момент инерции должен быть положительным")
        
        if C < 0:
            raise ValueError("Коэффициент линейного демпфирования не может быть отрицательным")
        
        if B < 0:
            raise ValueError("Коэффициент углового демпфирования не может быть отрицательным")
        
        self.velocity = velocity
        self.omega = omega
        self.m = m
        self.J = J
        self.C = C
        self.B = B
        self.dt = dt
        
        # Начальные состояния
        if v_old is None:
            self.v_old = np.zeros(2)
        else:
            self.v_old = np.asarray(v_old)
            if self.v_old.shape != (2,):
                raise ValueError("v_old должен быть вектором размера 2")
        
        self.omega_old = omega_old
        
        # Эффективные коэффициенты для неявной схемы
        if dt is not None:
            if dt <= 0:
                raise ValueError("Шаг по времени должен быть положительным")
            # Поступательное движение: (m/dt + C)*v_new = m/dt*v_old + F
            self.C_eff_linear = m / dt + C
            # Вращательное движение: (J/dt + B)*ω_new = J/dt*ω_old + τ
            self.C_eff_angular = J / dt + B
        else:
            # Статический анализ
            self.C_eff_linear = C if C > 0 else 1e-10
            self.C_eff_angular = B if B > 0 else 1e-10
    
    def get_variables(self) -> List[Variable]:
        return [self.velocity, self.omega]
    
    def contribute_to_A(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Добавить эффективные коэффициенты на диагональ
        """
        # Поступательное движение (2 степени свободы)
        v_indices = index_map[self.velocity]
        for idx in v_indices:
            A[idx, idx] += self.C_eff_linear
        
        # Вращательное движение (1 степень свободы)
        omega_idx = index_map[self.omega][0]
        A[omega_idx, omega_idx] += self.C_eff_angular
    
    def contribute_to_b(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Добавить вклад от предыдущего состояния
        """
        if self.dt is None:
            return
        
        # Инерционный член от предыдущей скорости
        v_indices = index_map[self.velocity]
        for i, idx in enumerate(v_indices):
            b[idx] += (self.m / self.dt) * self.v_old[i]
        
        # Инерционный член от предыдущей угловой скорости
        omega_idx = index_map[self.omega][0]
        b[omega_idx] += (self.J / self.dt) * self.omega_old
    
    def update_state(self, v_new: np.ndarray, omega_new: float):
        """
        Обновить состояние после шага по времени
        
        Args:
            v_new: Новая линейная скорость [vx, vy]
            omega_new: Новая угловая скорость [рад/с]
        """
        self.v_old = np.asarray(v_new)
        self.omega_old = omega_new
    
    def get_kinetic_energy(self, v: np.ndarray = None, omega: float = None) -> float:
        """
        Вычислить полную кинетическую энергию
        
        Returns:
            E_kin = (1/2)*m*v² + (1/2)*J*ω² [Дж]
        """
        if v is None:
            v = self.v_old
        if omega is None:
            omega = self.omega_old
        
        v = np.asarray(v)
        E_trans = 0.5 * self.m * np.dot(v, v)  # поступательная энергия
        E_rot = 0.5 * self.J * omega**2         # вращательная энергия
        
        return E_trans + E_rot
    
    def get_linear_momentum(self, v: np.ndarray = None) -> np.ndarray:
        """
        Вычислить линейный импульс
        
        Returns:
            p = m*v [кг·м/с]
        """
        if v is None:
            v = self.v_old
        return self.m * np.asarray(v)
    
    def get_angular_momentum(self, omega: float = None) -> float:
        """
        Вычислить угловой момент
        
        Returns:
            L = J*ω [кг·м²/с]
        """
        if omega is None:
            omega = self.omega_old
        return self.J * omega


class ForceVector2D(Contribution):
    """
    Векторная сила, приложенная к твердому телу (2D).
    
    Применяется к переменной velocity размера 2.
    """
    
    def __init__(self, velocity: Variable, force: np.ndarray):
        """
        Args:
            velocity: Переменная скорости (размер 2)
            force: Приложенная сила [Fx, Fy] [Н]
        """
        if velocity.size != 2:
            raise ValueError("velocity должна иметь размер 2")
        
        self.velocity = velocity
        self.force = np.asarray(force)
        if self.force.shape != (2,):
            raise ValueError("force должна быть вектором размера 2")
    
    def get_variables(self) -> List[Variable]:
        return [self.velocity]
    
    def contribute_to_A(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        pass
    
    def contribute_to_b(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        indices = index_map[self.velocity]
        for i, idx in enumerate(indices):
            b[idx] += self.force[i]
    
    def get_power(self, v: np.ndarray) -> float:
        """
        Вычислить мощность источника силы
        
        Returns:
            Мощность P = F·v [Вт]
        """
        return np.dot(self.force, v)


# ============================================================================
# Constraints (кинематические связи)
# ============================================================================


class RevoluteJoint2D(Constraint):
    """
    Вращательный шарнир (revolute joint) для твердого тела.
    
    Фиксирует точку на теле в пространстве, разрешая только вращение вокруг этой точки.
    
    Кинематическая связь:
    - Скорость точки крепления должна быть нулевой: v_cm + ω × r = 0
    - В 2D: v_cm + [-ω*ry, ω*rx] = 0
    
    где:
    - v_cm: скорость центра масс тела [vx, vy]
    - ω: угловая скорость тела
    - r: вектор от центра масс к точке крепления [rx, ry]
    
    Реализуется через множители Лагранжа (точное удовлетворение связи).
    """
    
    def __init__(self,
                 velocity: Variable,     # скорость центра масс [vx, vy]
                 omega: Variable,        # угловая скорость ω
                 r: np.ndarray):         # вектор от ЦМ к точке крепления [rx, ry]
        """
        Args:
            velocity: Переменная скорости центра масс (размер 2)
            omega: Переменная угловой скорости (размер 1)
            r: Вектор от центра масс к точке шарнира [rx, ry] [м]
        """
        if velocity.size != 2:
            raise ValueError("velocity должна иметь размер 2")
        
        if omega.size != 1:
            raise ValueError("omega должна быть скаляром")
        
        self.velocity = velocity
        self.omega = omega
        self.r = np.asarray(r)
        
        if self.r.shape != (2,):
            raise ValueError("r должен быть вектором размера 2")
        
        self.rx = r[0]
        self.ry = r[1]
        
        # Матрица коэффициентов связи
        # Связь: vx - ω*ry = 0
        #        vy + ω*rx = 0
        #
        # В матричной форме: C * [vx, vy, ω]^T = 0
        # C = [[1,  0, -ry],
        #      [0,  1,  rx]]
        
        self.C_velocity = np.array([
            [1.0, 0.0],   # коэффициенты при [vx, vy] для первого уравнения
            [0.0, 1.0]    # коэффициенты при [vx, vy] для второго уравнения
        ])
        
        self.C_omega = np.array([
            [-self.ry],  # коэффициент при ω для первого уравнения
            [ self.rx]   # коэффициент при ω для второго уравнения
        ])
    
    def get_variables(self) -> List[Variable]:
        return [self.velocity, self.omega]
    
    def get_n_constraints(self) -> int:
        """Два уравнения связи (vx и vy)"""
        return 2
    
    def contribute_to_C(self, C: np.ndarray, constraint_offset: int, 
                       index_map: Dict[Variable, List[int]]):
        """
        Добавить вклад в матрицу связей C
        """
        v_indices = index_map[self.velocity]
        omega_idx = index_map[self.omega][0]
        
        # Заполнить коэффициенты для vx и vy
        for i in range(2):  # две связи
            for j in range(2):  # два компонента velocity
                C[constraint_offset + i, v_indices[j]] += self.C_velocity[i, j]
            
            # Коэффициент при omega
            C[constraint_offset + i, omega_idx] += self.C_omega[i, 0]
    
    def contribute_to_d(self, d: np.ndarray, constraint_offset: int):
        """
        Правая часть связи (нулевая для фиксации точки)
        """
        # d[constraint_offset:constraint_offset+2] уже заполнено нулями
        pass
    
    def get_constraint_violation(self, v: np.ndarray, omega: float) -> np.ndarray:
        """
        Вычислить нарушение кинематической связи
        
        Returns:
            Вектор нарушения: v + ω × r [м/с]
        """
        v_pin = v + np.array([-omega * self.ry, omega * self.rx])
        return v_pin


class FixedPoint2D(Constraint):
    """
    Фиксация точки в пространстве (v = 0).
    
    Используется для полной фиксации скорости (например, заземление).
    """
    
    def __init__(self, velocity: Variable, target: np.ndarray = None):
        """
        Args:
            velocity: Переменная скорости (размер 2)
            target: Целевая скорость [vx, vy] (по умолчанию [0, 0])
        """
        if velocity.size != 2:
            raise ValueError("velocity должна иметь размер 2")
        
        self.velocity = velocity
        
        if target is None:
            self.target = np.zeros(2)
        else:
            self.target = np.asarray(target)
            if self.target.shape != (2,):
                raise ValueError("target должен быть вектором размера 2")
    
    def get_variables(self) -> List[Variable]:
        return [self.velocity]
    
    def get_n_constraints(self) -> int:
        """Два уравнения связи (vx = 0, vy = 0)"""
        return 2
    
    def contribute_to_C(self, C: np.ndarray, constraint_offset: int, 
                       index_map: Dict[Variable, List[int]]):
        """
        Матрица связи: единичная матрица [[1, 0], [0, 1]]
        """
        v_indices = index_map[self.velocity]
        
        C[constraint_offset, v_indices[0]] += 1.0      # vx = target[0]
        C[constraint_offset + 1, v_indices[1]] += 1.0  # vy = target[1]
    
    def contribute_to_d(self, d: np.ndarray, constraint_offset: int):
        """
        Правая часть: целевая скорость
        """
        d[constraint_offset:constraint_offset + 2] += self.target
