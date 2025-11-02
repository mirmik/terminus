#!/usr/bin/env python3
"""
Электромеханические элементы для связи электрических и механических систем.

Реализованы элементы, которые связывают электрическую и механическую домены:
- Двигатель постоянного тока (DC Motor)
- Генератор
- Линейный актуатор (соленоид)

Механические элементы (инерция, пружины и т.д.) находятся в модуле multibody.py
"""

import numpy as np
from typing import List, Dict
from .assembler import Contribution, Variable
from .multibody2d import RotationalInertia2D, TorqueSource2D  # импортируем из multibody2d


class DCMotor(Contribution):
    """
    Двигатель постоянного тока (упрощенная модель).
    
    Связывает электрическую и механическую системы через:
    - Электрическую цепь: V = R*I + L*dI/dt + K_e*ω (ЭДС противодействия)
    - Механическую часть: τ_motor = K_t*I (электромагнитный момент)
    
    Переменные:
    - V_plus, V_minus: электрические узлы (потенциалы)
    - omega: угловая скорость вала [рад/с]
    
    Параметры:
    - R: сопротивление обмотки [Ом]
    - L: индуктивность обмотки [Гн]
    - K_e: константа ЭДС [В/(рад/с)]
    - K_t: константа момента [Н·м/А]
    
    Примечание: в идеальном двигателе K_e = K_t
    """
    
    def __init__(self,
                 V_plus: Variable,      # положительный электрический вывод
                 V_minus: Variable,     # отрицательный электрический вывод
                 omega: Variable,       # угловая скорость (скаляр)
                 R: float,              # сопротивление [Ом]
                 L: float,              # индуктивность [Гн]
                 K_e: float,            # константа ЭДС [В/(рад/с)]
                 K_t: float,            # константа момента [Н·м/А]
                 dt: float = None,      # шаг по времени [с]
                 I_old: float = 0.0):   # ток на предыдущем шаге [А]
        """
        Args:
            V_plus: Переменная потенциала положительного вывода
            V_minus: Переменная потенциала отрицательного вывода
            omega: Переменная угловой скорости
            R: Сопротивление обмотки [Ом]
            L: Индуктивность обмотки [Гн]
            K_e: Константа ЭДС [В/(рад/с)]
            K_t: Константа момента [Н·м/А]
            dt: Шаг по времени для динамического анализа [с]
            I_old: Ток через обмотку на предыдущем шаге [А]
        """
        if V_plus.size != 1 or V_minus.size != 1 or omega.size != 1:
            raise ValueError("Все переменные должны быть скалярами")
        
        if R <= 0 or L <= 0:
            raise ValueError("R и L должны быть положительными")
        
        self.V_plus = V_plus
        self.V_minus = V_minus
        self.omega = omega
        self.R = R
        self.L = L
        self.K_e = K_e
        self.K_t = K_t
        self.dt = dt
        self.I_old = I_old
        
        # Для динамического анализа
        if dt is not None:
            if dt <= 0:
                raise ValueError("Шаг по времени должен быть положительным")
            # Эффективное сопротивление для неявной схемы
            self.R_eff = R + L / dt
            self.G_eff = 1.0 / self.R_eff
        else:
            # Статический анализ (только R)
            self.R_eff = R
            self.G_eff = 1.0 / R
        
        # Текущий ток (будет обновляться)
        self.I_current = I_old
    
    def get_variables(self) -> List[Variable]:
        return [self.V_plus, self.V_minus, self.omega]
    
    def contribute_to_A(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Добавить вклад в матрицу системы
        
        Связь электрической и механической частей:
        1. Электрическая часть: I = G_eff*(V+ - V- - K_e*ω)
        2. Механическая часть: τ_motor = K_t*I добавляется в уравнение для ω
        
        Система уравнений:
        Электрическое: Σ I_in_node = 0, где I = G*(V-V'-K_e*ω)
        Механическое: C_eff*ω = τ_motor + τ_other, где τ_motor = K_t*I
        
        Подставляя I:
        Механическое: C_eff*ω = K_t*G*(V+-V--K_e*ω) + τ_other
                      C_eff*ω + K_t*G*K_e*ω = K_t*G*(V+-V-) + τ_other
                      (C_eff + K_t*G*K_e)*ω - K_t*G*V+ + K_t*G*V- = τ_other
        
        Перепишем в форме A*x = b:
        Row для ω: ... + (K_t*G*K_e)*ω - (K_t*G)*V+ + (K_t*G)*V- = 0 (τ пойдет в b)
        """
        idx_plus = index_map[self.V_plus][0]
        idx_minus = index_map[self.V_minus][0]
        idx_omega = index_map[self.omega][0]
        
        G = self.G_eff
        
        # === Электрическая часть: проводимость между V+ и V- ===
        A[idx_plus, idx_plus] += G
        A[idx_plus, idx_minus] -= G
        A[idx_minus, idx_plus] -= G
        A[idx_minus, idx_minus] += G
        
        # === Связь 1: ЭДС противодействия влияет на ток ===
        # Ток зависит от ω: I = G*(V - K_e*ω)
        # Это влияет на баланс токов в узлах
        A[idx_plus, idx_omega] -= G * self.K_e
        A[idx_minus, idx_omega] += G * self.K_e
        
        # === Связь 2: Момент двигателя влияет на угловую скорость ===
        # Добавляем коэффициенты в строку для omega
        A[idx_omega, idx_omega] += self.K_t * G * self.K_e  # связь ω с собой
        A[idx_omega, idx_plus] -= self.K_t * G  # связь ω с V+
        A[idx_omega, idx_minus] += self.K_t * G  # связь ω с V-
    
    def contribute_to_b(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Добавить вклад в правую часть
        
        Учитывает историю тока для динамического анализа.
        
        ВАЖНО: Момент двигателя добавляется в уравнение для omega.
        Это связывает электрическую и механическую части.
        """
        idx_plus = index_map[self.V_plus][0]
        idx_minus = index_map[self.V_minus][0]
        idx_omega = index_map[self.omega][0]
        
        if self.dt is not None:
            # Вклад от предыдущего тока в электрическую часть
            I_history_term = (self.L / self.dt) * self.I_old
            
            # Эквивалентный источник тока
            b[idx_plus] -= self.G_eff * I_history_term
            b[idx_minus] += self.G_eff * I_history_term
        
        # Момент двигателя: τ = K_t * G * (V+ - V-)
        # НО V+ и V- - это переменные, поэтому вклад идет через матрицу A, а не b
        # Вместо этого добавим постоянный вклад: 0 (нет постоянного момента)
    
    def get_current(self, V_plus: float, V_minus: float, omega: float) -> float:
        """
        Вычислить ток через обмотку двигателя
        
        Args:
            V_plus: Потенциал положительного вывода [В]
            V_minus: Потенциал отрицательного вывода [В]
            omega: Угловая скорость [рад/с]
        
        Returns:
            Ток I [А]
        """
        V = V_plus - V_minus
        EMF = self.K_e * omega  # ЭДС противодействия
        
        if self.dt is None:
            # Статический анализ
            I = (V - EMF) / self.R
        else:
            # Динамический анализ
            I = self.G_eff * (V - EMF) + (self.L / self.dt / self.R_eff) * self.I_old
        
        return I
    
    def get_torque(self, I: float = None, V_plus: float = None, 
                   V_minus: float = None, omega: float = None) -> float:
        """
        Вычислить электромагнитный момент двигателя
        
        Args:
            I: Ток через обмотку [А] (если None, вычисляется из напряжений)
            V_plus, V_minus, omega: для вычисления тока, если I не задан
        
        Returns:
            Момент τ [Н·м]
        """
        if I is None:
            if V_plus is None or V_minus is None or omega is None:
                raise ValueError("Нужно задать либо I, либо (V_plus, V_minus, omega)")
            I = self.get_current(V_plus, V_minus, omega)
        
        return self.K_t * I
    
    def update_state(self, V_plus: float, V_minus: float, omega: float):
        """
        Обновить состояние двигателя после шага по времени
        """
        self.I_current = self.get_current(V_plus, V_minus, omega)
        self.I_old = self.I_current


# Вспомогательная функция для создания простой системы двигатель + нагрузка
def create_motor_system(V_source: float, R: float, L: float, 
                       K_e: float, K_t: float, J: float, B: float,
                       dt: float):
    """
    Создать простую систему: источник напряжения -> двигатель -> инерция
    
    Args:
        V_source: Напряжение питания [В]
        R: Сопротивление обмотки [Ом]
        L: Индуктивность обмотки [Гн]
        K_e: Константа ЭДС [В/(рад/с)]
        K_t: Константа момента [Н·м/А]
        J: Момент инерции нагрузки [кг·м²]
        B: Коэффициент трения [Н·м·с]
        dt: Шаг по времени [с]
    
    Returns:
        (elements, variables): список элементов и переменных
    """
    from .electrical import VoltageSource, Ground
    
    # Создать переменные
    v_plus = Variable("V+", 1)
    v_gnd = Variable("GND", 1)
    omega = Variable("omega", 1)
    
    # Электрическая часть
    v_src = VoltageSource(v_plus, v_gnd, V_source)
    ground = Ground(v_gnd)
    
    # Двигатель
    motor = DCMotor(v_plus, v_gnd, omega, R, L, K_e, K_t, dt=dt)
    
    # Механическая часть
    inertia = RotationalInertia2D(omega, J, B, dt=dt)
    
    elements = [v_src, ground, motor, inertia]
    variables = [v_plus, v_gnd, omega]
    
    return elements, variables
