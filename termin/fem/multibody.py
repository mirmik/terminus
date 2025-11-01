#!/usr/bin/env python3
"""
Элементы многотельной механики для вращательного движения.

Реализованы элементы для моделирования вращательных механических систем:
- Вращательная инерция с демпфированием
- Источник момента
- Вращательная пружина (торсион)
- Фиксированная скорость (граничное условие)
- Вращательный демпфер
"""

import numpy as np
from typing import List, Dict
from .assembler import Contribution, Variable


class RotationalInertia(Contribution):
    """
    Вращательная инерция с демпфированием.
    
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


class TorqueSource(Contribution):
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


class RotationalSpring(Contribution):
    """
    Вращательная пружина (торсион).
    
    Связывает две угловые скорости через упругий момент:
    τ = K * (θ1 - θ2)
    
    Для работы нужны угловые скорости, момент связан с разностью углов.
    В динамике: τ = K * ∫(ω1 - ω2)dt
    
    Упрощенный вариант: работает как демпфер между скоростями.
    """
    
    def __init__(self,
                 omega1: Variable,  # первая угловая скорость
                 omega2: Variable,  # вторая угловая скорость
                 K: float):         # жесткость пружины [Н·м/рад]
        """
        Args:
            omega1: Переменная первой угловой скорости
            omega2: Переменная второй угловой скорости
            K: Жесткость пружины [Н·м/рад]
        """
        if omega1.size != 1 or omega2.size != 1:
            raise ValueError("Переменные должны быть скалярами")
        
        if K <= 0:
            raise ValueError("Жесткость должна быть положительной")
        
        self.omega1 = omega1
        self.omega2 = omega2
        self.K = K
    
    def get_variables(self) -> List[Variable]:
        return [self.omega1, self.omega2]
    
    def contribute_to_A(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Упругая связь между скоростями (действует как демпфер)
        """
        idx1 = index_map[self.omega1][0]
        idx2 = index_map[self.omega2][0]
        
        K = self.K
        
        # Симметричная матрица связи
        A[idx1, idx1] += K
        A[idx1, idx2] -= K
        A[idx2, idx1] -= K
        A[idx2, idx2] += K
    
    def contribute_to_b(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Пружина без начального напряжения не дает вклад в правую часть
        """
        pass


class RotationalDamper(Contribution):
    """
    Вращательный демпфер (вязкое трение между двумя телами).
    
    Создает момент, пропорциональный разности скоростей:
    τ = B * (ω1 - ω2)
    
    Аналог линейного демпфера, но для вращательного движения.
    """
    
    def __init__(self,
                 omega1: Variable,  # первая угловая скорость
                 omega2: Variable,  # вторая угловая скорость
                 B: float):         # коэффициент демпфирования [Н·м·с]
        """
        Args:
            omega1: Переменная первой угловой скорости
            omega2: Переменная второй угловой скорости
            B: Коэффициент вязкого трения [Н·м·с]
        """
        if omega1.size != 1 or omega2.size != 1:
            raise ValueError("Переменные должны быть скалярами")
        
        if B <= 0:
            raise ValueError("Коэффициент демпфирования должен быть положительным")
        
        self.omega1 = omega1
        self.omega2 = omega2
        self.B = B
    
    def get_variables(self) -> List[Variable]:
        return [self.omega1, self.omega2]
    
    def contribute_to_A(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Демпфирование между двумя телами
        """
        idx1 = index_map[self.omega1][0]
        idx2 = index_map[self.omega2][0]
        
        B = self.B
        
        # Симметричная матрица связи
        A[idx1, idx1] += B
        A[idx1, idx2] -= B
        A[idx2, idx1] -= B
        A[idx2, idx2] += B
    
    def contribute_to_b(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Демпфер не дает вклад в правую часть
        """
        pass
    
    def get_dissipated_power(self, omega1: float, omega2: float) -> float:
        """
        Вычислить рассеиваемую мощность
        
        Args:
            omega1: Скорость первого тела [рад/с]
            omega2: Скорость второго тела [рад/с]
        
        Returns:
            Рассеиваемая мощность P = B*(ω1-ω2)² [Вт]
        """
        delta_omega = omega1 - omega2
        return self.B * delta_omega**2


class FixedRotation(Contribution):
    """
    Фиксированная угловая скорость (граничное условие).
    
    Аналог Ground для механической системы.
    Используется для задания кинематических граничных условий.
    """
    
    def __init__(self, omega: Variable, value: float = 0.0):
        """
        Args:
            omega: Переменная угловой скорости
            value: Фиксированное значение [рад/с]
        """
        if omega.size != 1:
            raise ValueError("omega должна быть скаляром")
        
        self.omega = omega
        self.value = value
        self.penalty = 1e10
    
    def get_variables(self) -> List[Variable]:
        return [self.omega]
    
    def contribute_to_A(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Большое число на диагонали
        """
        idx = index_map[self.omega][0]
        A[idx, idx] += self.penalty
    
    def contribute_to_b(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Правая часть = penalty * value
        """
        idx = index_map[self.omega][0]
        b[idx] += self.penalty * self.value


class LinearMass(Contribution):
    """
    Линейная масса (поступательное движение).
    
    Уравнение: m*dv/dt = Σ F - c*v
    
    где:
    - m: масса [кг]
    - v: скорость [м/с]
    - c: коэффициент вязкого сопротивления [Н·с/м]
    - F: внешние силы [Н]
    """
    
    def __init__(self,
                 velocity: Variable,  # линейная скорость
                 m: float,            # масса [кг]
                 c: float = 0.0,      # коэффициент демпфирования [Н·с/м]
                 dt: float = None,    # шаг по времени [с]
                 v_old: float = 0.0): # скорость на предыдущем шаге
        """
        Args:
            velocity: Переменная линейной скорости
            m: Масса [кг]
            c: Коэффициент вязкого сопротивления [Н·с/м]
            dt: Шаг по времени [с]
            v_old: Скорость на предыдущем шаге [м/с]
        """
        if velocity.size != 1:
            raise ValueError("velocity должна быть скаляром")
        
        if m <= 0:
            raise ValueError("Масса должна быть положительной")
        
        if c < 0:
            raise ValueError("Коэффициент демпфирования не может быть отрицательным")
        
        self.velocity = velocity
        self.m = m
        self.c = c
        self.dt = dt
        self.v_old = v_old
        
        if dt is not None:
            if dt <= 0:
                raise ValueError("Шаг по времени должен быть положительным")
            self.C_eff = m / dt + c
        else:
            self.C_eff = c if c > 0 else 1e-10
    
    def get_variables(self) -> List[Variable]:
        return [self.velocity]
    
    def contribute_to_A(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        idx = index_map[self.velocity][0]
        A[idx, idx] += self.C_eff
    
    def contribute_to_b(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        if self.dt is None:
            return
        
        idx = index_map[self.velocity][0]
        b[idx] += (self.m / self.dt) * self.v_old
    
    def update_state(self, v_new: float):
        """
        Обновить состояние после шага по времени
        """
        self.v_old = v_new
    
    def get_kinetic_energy(self, v: float = None) -> float:
        """
        Вычислить кинетическую энергию
        
        Returns:
            Кинетическая энергия E_k = (1/2)*m*v² [Дж]
        """
        if v is None:
            v = self.v_old
        return 0.5 * self.m * v**2


class ForceSource(Contribution):
    """
    Источник силы (внешняя сила, приложенная к телу).
    """
    
    def __init__(self, velocity: Variable, force: float):
        """
        Args:
            velocity: Переменная скорости
            force: Приложенная сила [Н]
        """
        if velocity.size != 1:
            raise ValueError("velocity должна быть скаляром")
        
        self.velocity = velocity
        self.force = force
    
    def get_variables(self) -> List[Variable]:
        return [self.velocity]
    
    def contribute_to_A(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        pass
    
    def contribute_to_b(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        idx = index_map[self.velocity][0]
        b[idx] += self.force
    
    def get_power(self, v: float) -> float:
        """
        Вычислить мощность источника силы
        
        Returns:
            Мощность P = F*v [Вт]
        """
        return self.force * v


class LinearSpring(Contribution):
    """
    Линейная пружина между двумя массами.
    
    Создает силу, пропорциональную разности скоростей (в упрощенной модели).
    """
    
    def __init__(self,
                 v1: Variable,  # первая скорость
                 v2: Variable,  # вторая скорость
                 k: float):     # жесткость [Н/м]
        """
        Args:
            v1: Переменная первой скорости
            v2: Переменная второй скорости
            k: Жесткость пружины [Н/м]
        """
        if v1.size != 1 or v2.size != 1:
            raise ValueError("Переменные должны быть скалярами")
        
        if k <= 0:
            raise ValueError("Жесткость должна быть положительной")
        
        self.v1 = v1
        self.v2 = v2
        self.k = k
    
    def get_variables(self) -> List[Variable]:
        return [self.v1, self.v2]
    
    def contribute_to_A(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        idx1 = index_map[self.v1][0]
        idx2 = index_map[self.v2][0]
        
        k = self.k
        
        A[idx1, idx1] += k
        A[idx1, idx2] -= k
        A[idx2, idx1] -= k
        A[idx2, idx2] += k
    
    def contribute_to_b(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        pass
