#!/usr/bin/env python3
"""
Элементы многотельной механики для плоского (2D) движения.

Реализованы элементы для моделирования планарных механических систем:

Contributions (физические элементы):
- RotationalInertia2D: вращательная инерция с демпфированием
- TorqueSource2D: источник момента
- RigidBody2D: твердое тело (3 DOF: vx, vy, ω)
- ForceOnBody2D: обобщенная нагрузка (сила и/или момент) на тело
- ForceVector2D: векторная сила на тело (устаревший)

Constraints (кинематические связи):
- FixedRevoluteJoint2D: шарнир с фиксацией в пространстве (заземление)
- TwoBodyRevoluteJoint2D: вращательный шарнир между двумя телами
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
                 assembler=None):
        """
        Args:
            omega: Переменная угловой скорости
            J: Момент инерции [кг·м²]
            B: Коэффициент вязкого трения [Н·м·с]
            dt: Шаг по времени [с]
            assembler: MatrixAssembler для автоматической регистрации
        """
        if omega.size != 1:
            raise ValueError("omega должна быть скаляром")
        
        if J <= 0:
            raise ValueError("Момент инерции должен быть положительным")
        
        if B < 0:
            raise ValueError("Коэффициент демпфирования не может быть отрицательным")
        
        super().__init__([omega], assembler)
        
        self.omega = omega
        self.J = J
        self.B = B
        self.dt = dt
        
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
    
    def contribute_to_mass(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Добавить эффективный коэффициент в диагональ
        """
        idx = index_map[self.omega][0]
        A[idx, idx] += self.C_eff

    def contribute_to_damping(self, C, index_map):
        idx = index_map[self.omega][0]
        C[idx, idx] += self.B
    
    def contribute_to_load(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Добавить вклад от предыдущего состояния
        """
        if self.dt is None:
            return
        
        idx = index_map[self.omega][0]
        # Инерционный член от предыдущего шага
        b[idx] += (self.J / self.dt) * self.omega.value
    
    def get_kinetic_energy(self, omega: float = None) -> float:
        """
        Вычислить кинетическую энергию вращения
        
        Args:
            omega: Угловая скорость [рад/с] (если None, используется текущая)
        
        Returns:
            Кинетическая энергия E_k = (1/2)*J*ω² [Дж]
        """
        if omega is None:
            omega = self.omega.value
        return 0.5 * self.J * omega**2


class TorqueSource2D(Contribution):
    """
    Источник момента (внешний момент, приложенный к вращающемуся телу).
    Аналог источника тока для механической системы.
    """
    
    def __init__(self, omega: Variable, torque: float, assembler=None):
        """
        Args:
            omega: Переменная угловой скорости
            torque: Приложенный момент [Н·м] (положительный = ускорение)
            assembler: MatrixAssembler для автоматической регистрации
        """
        if omega.size != 1:
            raise ValueError("omega должна быть скаляром")
        
        super().__init__([omega], assembler)
        
        self.omega = omega
        self.torque = torque
    
    def contribute_to_mass(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Источник момента не влияет на матрицу
        """
        pass
    
    def contribute_to_load(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
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
                 m: float,                # масса [кг]
                 J: float,                # момент инерции [кг·м²]
                 C: float = 0.0,          # коэффициент линейного демпфирования [Н·с/м]
                 B: float = 0.0,          # коэффициент углового демпфирования [Н·м·с]
                 dt: float = None,        # шаг по времени [с]
                 velocity: Variable = None,  # переменная линейной скорости [vx, vy]
                 omega: Variable = None,      # переменная угловой скорости ω
                 assembler=None):             # ассемблер для автоматической регистрации
        """
        Args:
            m: Масса [кг]
            J: Момент инерции [кг·м²]
            C: Коэффициент вязкого сопротивления для поступательного движения [Н·с/м]
            B: Коэффициент вязкого трения для вращательного движения [Н·м·с]
            dt: Шаг по времени [с]
            velocity: Переменная линейной скорости (размер 2)
            omega: Переменная угловой скорости (размер 1)
            assembler: MatrixAssembler для автоматической регистрации переменных
        """
        if velocity is None:
            velocity = Variable("v_body", size=2)

        if omega is None:
            omega = Variable("omega_body", size=1)

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
        
        super().__init__([velocity, omega], assembler)
        
        self.velocity = velocity
        self.omega = omega
        self.m = m
        self.J = J
        self.C = C
        self.B = B
        self.dt = dt
        
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

    def set_values(self, v: np.ndarray, w: float):
        """
        Установить текущее состояние тела

        Args:
            v: Линейная скорость [vx, vy]
            w: Угловая скорость [рад/с]
        """
        self.velocity.set_value(v)
        self.omega.set_value(w)
    
    def contribute_to_mass(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
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
    
    def contribute_to_load(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Добавить вклад от предыдущего состояния
        """
        if self.dt is None:
            return
        
        # Инерционный член от предыдущей скорости
        v_indices = index_map[self.velocity]
        for i, idx in enumerate(v_indices):
            b[idx] += (self.m / self.dt) * self.velocity.value[i]
        
        # Инерционный член от предыдущей угловой скорости
        omega_idx = index_map[self.omega][0]
        b[omega_idx] += (self.J / self.dt) * self.omega.value
    
    def get_kinetic_energy(self, v: np.ndarray = None, omega: float = None) -> float:
        """
        Вычислить полную кинетическую энергию
        
        Returns:
            E_kin = (1/2)*m*v² + (1/2)*J*ω² [Дж]
        """
        if v is None:
            v = self.velocity.value
        if omega is None:
            omega = self.omega.value

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
            v = self.velocity.value
        return self.m * np.asarray(v)
    
    def get_angular_momentum(self, omega: float = None) -> float:
        """
        Вычислить угловой момент
        
        Returns:
            L = J*ω [кг·м²/с]
        """
        if omega is None:
            omega = self.omega.value
        return self.J * omega


class ForceOnBody2D(Contribution):
    """
    Обобщенная нагрузка на твердое тело (сила и/или момент).
    
    Применяет силу к центру масс и/или момент к телу.
    Более явная семантика по сравнению с ForceVector2D и TorqueSource2D.
    """
    
    def __init__(self, 
                 body: 'RigidBody2D',
                 force: np.ndarray = None,
                 torque: float = None,
                 assembler=None):
        """
        Args:
            body: Твердое тело, к которому применяется нагрузка
            force: Сила [Fx, Fy] [Н], приложенная к центру масс (опционально)
            torque: Момент [Н·м] вокруг центра масс (опционально)
            assembler: MatrixAssembler для автоматической регистрации переменных
        """
        if force is None and torque is None:
            raise ValueError("Должна быть указана хотя бы одна нагрузка: force или torque")
        
        self.body = body
        self.velocity = body.velocity
        self.omega = body.omega
        
        # Сила
        if force is not None:
            self.force = np.asarray(force)
            if self.force.shape != (2,):
                raise ValueError("force должна быть вектором размера 2")
        else:
            self.force = None
        
        # Момент
        self.torque = torque
        
        # Определяем список переменных
        vars_list = []
        if self.force is not None:
            vars_list.append(self.velocity)
        if self.torque is not None:
            vars_list.append(self.omega)
        
        super().__init__(vars_list, assembler)
    
    def contribute_to_mass(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        """Нагрузка не влияет на матрицу"""
        pass
    
    def contribute_to_load(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        """Добавить нагрузку в правую часть"""
        # Добавляем силу
        if self.force is not None:
            v_indices = index_map[self.velocity]
            for i, idx in enumerate(v_indices):
                b[idx] += self.force[i]
        
        # Добавляем момент
        if self.torque is not None:
            omega_idx = index_map[self.omega][0]
            b[omega_idx] += self.torque
    
    def get_power(self) -> float:
        """
        Вычислить мощность нагрузки
        
        Returns:
            Мощность P = F·v + τ·ω [Вт]
        """
        power = 0.0
        if self.force is not None:
            power += np.dot(self.force, self.velocity.value)
        if self.torque is not None:
            power += self.torque * self.omega.value
        return power


class ForceVector2D(Contribution):
    """
    Векторная сила, приложенная к твердому телу (2D).
    Применяется к переменной velocity размера 2.
    """
    
    def __init__(self, velocity: Variable, force: np.ndarray, assembler=None):
        """
        Args:
            velocity: Переменная скорости (размер 2)
            force: Приложенная сила [Fx, Fy] [Н]
            assembler: MatrixAssembler для автоматической регистрации переменных
        """
        if velocity.size != 2:
            raise ValueError("velocity должна иметь размер 2")
        
        self.velocity = velocity
        self.force = np.asarray(force)
        if self.force.shape != (2,):
            raise ValueError("force должна быть вектором размера 2")
        
        super().__init__([velocity], assembler)
    
    def contribute_to_mass(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        pass
    
    def contribute_to_load(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
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


class FixedRevoluteJoint2D(Constraint):
    """
    Вращательный шарнир с фиксацией в пространстве (ground revolute joint).
    
    Фиксирует точку на теле в пространстве, разрешая только вращение вокруг этой точки.
    Эквивалентно присоединению тела к неподвижному основанию через шарнир.
    
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
                 body: 'RigidBody2D',    # твердое тело
                 r: np.ndarray,          # вектор от ЦМ к точке крепления [rx, ry]
                 assembler=None):        # ассемблер для автоматической регистрации
        """
        Args:
            body: Твердое тело, к которому применяется связь
            r: Вектор от центра масс к точке шарнира [rx, ry] [м]
            assembler: MatrixAssembler для автоматической регистрации переменных
        """
        self.body = body
        self.r = np.asarray(r)

        self.lambdas = Variable("lambda_fixed_revolute", size=2)  # два множителя Лагранжа
        
        if self.r.shape != (2,):
            raise ValueError("r должен быть вектором размера 2")
        
        super().__init__(
            [self.body.velocity, self.body.omega], 
            holonomic_lambdas=[self.lambdas], 
            nonholonomic_lambdas=[],
            assembler=assembler)
        
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
    
    def update_r(self, r: np.ndarray):
        """
        Обновить вектор от центра масс к точке шарнира.
        
        Args:
            r: Новый вектор [rx, ry] [м]
        """
        self.r = np.asarray(r)
        if self.r.shape != (2,):
            raise ValueError("r должен быть вектором размера 2")
        
        self.rx = r[0]
        self.ry = r[1]
        
        # Обновляем матрицу коэффициентов
        self.C_omega = np.array([
            [-self.ry],
            [ self.rx]
        ])
    
    def contribute_to_holonomic(self, C: np.ndarray,
                       index_map: Dict[Variable, List[int]]):
        """
        Добавить вклад в матрицу связей C
        """
        v_indices = index_map[self.body.velocity]
        omega_idx = index_map[self.body.omega][0]
        constr_indices = index_map[self.lambdas[0]]

        # Заполнить коэффициенты для vx и vy
        for i in range(2):  # две связи
            for j in range(2):  # два компонента velocity
                C[constr_indices[i], v_indices[j]] += self.C_velocity[i, j]
            
            # Коэффициент при omega
            C[constr_indices[i], omega_idx] += self.C_omega[i, 0]

    def contribute_to_holonomic_load(self, d: np.ndarray,  index_map: Dict[Variable, List[int]]):
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


class TwoBodyRevoluteJoint2D(Constraint):
    """
    Вращательный шарнир между двумя телами (two-body revolute joint).
    
    Соединяет два тела через точку, разрешая только относительное вращение.
    
    Кинематическая связь:
    - Скорости точек крепления на обоих телах должны совпадать:
      v1_cm + ω1 × r1 = v2_cm + ω2 × r2
    - В 2D: v1 + [-ω1*r1y, ω1*r1x] = v2 + [-ω2*r2y, ω2*r2x]
    
    Это даёт два уравнения связи (для x и y компонент).
    
    где:
    - v1_cm, v2_cm: скорости центров масс тел [vx, vy]
    - ω1, ω2: угловые скорости тел
    - r1, r2: векторы от центров масс к точке шарнира [rx, ry]
    
    Реализуется через множители Лагранжа.
    """
    
    def __init__(self,
                 body1: 'RigidBody2D',   # первое тело
                 body2: 'RigidBody2D',   # второе тело
                 r1: np.ndarray,         # вектор от ЦМ тела1 к точке шарнира
                 r2: np.ndarray,         # вектор от ЦМ тела2 к точке шарнира
                 assembler=None):
        """
        Args:
            body1: Первое твердое тело
            body2: Второе твердое тело
            r1: Вектор от центра масс body1 к точке шарнира [rx, ry] [м]
            r2: Вектор от центра масс body2 к точке шарнира [rx, ry] [м]
            assembler: MatrixAssembler для автоматической регистрации переменных
        """
        self.body1 = body1
        self.body2 = body2
        self.velocity1 = body1.velocity
        self.velocity2 = body2.velocity
        self.omega1 = body1.omega
        self.omega2 = body2.omega

        self.lambdas = Variable("lambda_two_body_revolute", size=2)  # два множителя Лагранжа
        
        self.r1 = np.asarray(r1)
        self.r2 = np.asarray(r2)
        
        if self.r1.shape != (2,):
            raise ValueError("r1 должен быть вектором размера 2")
        if self.r2.shape != (2,):
            raise ValueError("r2 должен быть вектором размера 2")

        super().__init__([self.velocity1, self.omega1, self.velocity2, self.omega2], [self.lambdas], assembler)

        self.r1x = r1[0]
        self.r1y = r1[1]
        self.r2x = r2[0]
        self.r2y = r2[1]
        
        # Матрица коэффициентов связи
        # Связь: v1x - ω1*r1y - v2x + ω2*r2y = 0
        #        v1y + ω1*r1x - v2y - ω2*r2x = 0
        #
        # В матричной форме: C * [v1x, v1y, ω1, v2x, v2y, ω2]^T = 0
        # C = [[1,  0, -r1y, -1,  0,  r2y],
        #      [0,  1,  r1x,  0, -1, -r2x]]
    
    def update_r(self, r1: np.ndarray, r2: np.ndarray):
        """
        Обновить векторы от центров масс к точке шарнира.
        
        Args:
            r1: Новый вектор для body1 [rx, ry] [м]
            r2: Новый вектор для body2 [rx, ry] [м]
        """
        self.r1 = np.asarray(r1)
        self.r2 = np.asarray(r2)
        
        if self.r1.shape != (2,):
            raise ValueError("r1 должен быть вектором размера 2")
        if self.r2.shape != (2,):
            raise ValueError("r2 должен быть вектором размера 2")
        
        self.r1x = r1[0]
        self.r1y = r1[1]
        self.r2x = r2[0]
        self.r2y = r2[1]
    
    def get_n_holonomic(self) -> int:
        """Два уравнения связи (x и y компоненты)"""
        return 2
    
    def contribute_to_holonomic(self, C: np.ndarray,
                       index_map: Dict[Variable, List[int]]):
        """
        Добавить вклад в матрицу связей C
        """
        v1_indices = index_map[self.velocity1]
        omega1_idx = index_map[self.omega1][0]
        v2_indices = index_map[self.velocity2]
        omega2_idx = index_map[self.omega2][0]

        constr_indices = index_map[self.lambdas[0]]
        
        # Первое уравнение: v1x - ω1*r1y - v2x + ω2*r2y = 0
        C[constr_indices[0], v1_indices[0]] += 1.0      # v1x
        C[constr_indices[0], omega1_idx] += -self.r1y   # -ω1*r1y
        C[constr_indices[0], v2_indices[0]] += -1.0     # -v2x
        C[constr_indices[0], omega2_idx] += self.r2y    # ω2*r2y
        
        # Второе уравнение: v1y + ω1*r1x - v2y - ω2*r2x = 0
        C[constr_indices[1], v1_indices[1]] += 1.0      # v1y
        C[constr_indices[1], omega1_idx] += self.r1x    # ω1*r1x
        C[constr_indices[1], v2_indices[1]] += -1.0     # -v2y
        C[constr_indices[1], omega2_idx] += -self.r2x   # -ω2*r2x
    
    def contribute_to_holonomic_load(self, d: np.ndarray, constraint_offset: int):
        """
        Правая часть связи (нулевая для совпадения скоростей точек)
        """
        # d[constraint_offset:constraint_offset+2] уже заполнено нулями
        pass
    
    def get_constraint_violation(self, v1: np.ndarray, omega1: float,
                                  v2: np.ndarray, omega2: float) -> np.ndarray:
        """
        Вычислить нарушение кинематической связи
        
        Returns:
            Вектор нарушения: (v1 + ω1 × r1) - (v2 + ω2 × r2) [м/с]
        """
        v1_pin = v1 + np.array([-omega1 * self.r1y, omega1 * self.r1x])
        v2_pin = v2 + np.array([-omega2 * self.r2y, omega2 * self.r2x])
        return v1_pin - v2_pin


class FixedPoint2D(Constraint):
    """
    Фиксация точки в пространстве (v = 0).
    
    Используется для полной фиксации скорости тела (например, заземление).
    """
    
    def __init__(self, body: 'RigidBody2D', target: np.ndarray = None, assembler=None):
        """
        Args:
            body: Твердое тело, к которому применяется связь
            target: Целевая скорость [vx, vy] (по умолчанию [0, 0])
            assembler: MatrixAssembler для автоматической регистрации переменных
        """
        self.body = body
        self.velocity = body.velocity
        
        if target is None:
            self.target = np.zeros(2)
        else:
            self.target = np.asarray(target)
            if self.target.shape != (2,):
                raise ValueError("target должен быть вектором размера 2")
        
        super().__init__([self.velocity], assembler)
    
    def get_n_constraints(self) -> int:
        """Два уравнения связи (vx = 0, vy = 0)"""
        return 2
    
    def contribute_to_damping(self, C: np.ndarray, constraint_offset: int, 
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
