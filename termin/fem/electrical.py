#!/usr/bin/env python3
"""
Простейшие конечные элементы для электротехники.

Реализованы классические элементы для анализа электрических цепей:
- Резистор (Resistor) - линейное сопротивление
- Конденсатор (Capacitor) - реактивное сопротивление (динамика)
- Катушка индуктивности (Inductor) - реактивное сопротивление (динамика)
- Источник напряжения (VoltageSource) - задает разность потенциалов
- Источник тока (CurrentSource) - задает ток через элемент

Элементы работают по аналогии с механическими:
- Узлы = потенциалы в узлах схемы
- "Жесткость" = проводимость (G = 1/R)
- Закон Кирхгофа ≈ баланс сил
"""

import numpy as np
from typing import List, Dict
from .assembler import Contribution, Variable


class Resistor(Contribution):
    """
    Резистор - линейный элемент.
    
    Закон Ома: U = I * R, или I = (V1 - V2) / R = (V1 - V2) * G
    где G = 1/R - проводимость
    
    Матрица проводимости (аналог матрицы жесткости):
    G_matrix = G * [[ 1, -1],
                    [-1,  1]]
    
    Уравнение: [I1]   = G * [[ 1, -1]] * [V1]
               [I2]         [[-1,  1]]   [V2]
    """
    
    def __init__(self, 
                 node1: Variable,  # потенциал узла 1
                 node2: Variable,  # потенциал узла 2
                 R: float):        # сопротивление [Ом]
        """
        Args:
            node1: Переменная потенциала первого узла (скаляр)
            node2: Переменная потенциала второго узла (скаляр)
            R: Сопротивление [Ом]
        """
        if node1.size != 1 or node2.size != 1:
            raise ValueError("Узлы должны быть скалярами (потенциалы)")
        
        if R <= 0:
            raise ValueError("Сопротивление должно быть положительным")
        
        self.node1 = node1
        self.node2 = node2
        self.R = R
        self.G = 1.0 / R  # проводимость
    
    def get_conductance_matrix(self) -> np.ndarray:
        """
        Матрица проводимости 2x2
        """
        G = self.G
        return G * np.array([
            [ 1, -1],
            [-1,  1]
        ])
    
    def get_variables(self) -> List[Variable]:
        return [self.node1, self.node2]
    
    def contribute_to_A(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Добавить матрицу проводимости в глобальную систему
        """
        G_matrix = self.get_conductance_matrix()
        
        # Получить глобальные индексы
        idx1 = index_map[self.node1][0]
        idx2 = index_map[self.node2][0]
        global_indices = [idx1, idx2]
        
        # Добавить в глобальную матрицу
        for i, gi in enumerate(global_indices):
            for j, gj in enumerate(global_indices):
                A[gi, gj] += G_matrix[i, j]
    
    def contribute_to_b(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Резистор без источников не вносит вклад в правую часть
        """
        pass
    
    def get_current(self, V1: float, V2: float) -> float:
        """
        Вычислить ток через резистор по закону Ома
        
        Args:
            V1: Потенциал узла 1 [В]
            V2: Потенциал узла 2 [В]
        
        Returns:
            Ток I [А] (от узла 1 к узлу 2)
        """
        return (V1 - V2) / self.R
    
    def get_power(self, V1: float, V2: float) -> float:
        """
        Вычислить рассеиваемую мощность
        
        Returns:
            Мощность P [Вт]
        """
        I = self.get_current(V1, V2)
        return I**2 * self.R


class Capacitor(Contribution):
    """
    Конденсатор - реактивный элемент (накапливает энергию в электрическом поле).
    
    Соотношение: I = C * dV/dt
    где V = V1 - V2 - напряжение на конденсаторе
    
    Для статического анализа (DC) конденсатор = разрыв (I = 0).
    Для динамического анализа используется дискретизация по времени.
    
    При использовании неявной схемы Эйлера:
    dV/dt ≈ (V_new - V_old) / dt
    
    Получаем: I = C * (V_new - V_old) / dt
    Или: C/dt * V_new = I + C/dt * V_old
    
    Эффективная проводимость: G_eff = C / dt
    """
    
    def __init__(self,
                 node1: Variable,
                 node2: Variable,
                 C: float,           # емкость [Ф]
                 dt: float = None,   # шаг по времени [с]
                 V_old: float = 0.0): # напряжение на предыдущем шаге
        """
        Args:
            node1: Переменная потенциала первого узла
            node2: Переменная потенциала второго узла
            C: Емкость [Ф]
            dt: Шаг по времени для динамического анализа [с] (None для статики)
            V_old: Напряжение на конденсаторе на предыдущем шаге [В]
        """
        if node1.size != 1 or node2.size != 1:
            raise ValueError("Узлы должны быть скалярами")
        
        if C <= 0:
            raise ValueError("Емкость должна быть положительной")
        
        self.node1 = node1
        self.node2 = node2
        self.C = C
        self.dt = dt
        self.V_old = V_old
        
        # Для динамического анализа вычислить эффективную проводимость
        if dt is not None:
            if dt <= 0:
                raise ValueError("Шаг по времени должен быть положительным")
            self.G_eff = C / dt
        else:
            self.G_eff = 0  # В статике конденсатор = разрыв
    
    def get_variables(self) -> List[Variable]:
        return [self.node1, self.node2]
    
    def contribute_to_A(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        В динамике: добавляем эффективную проводимость G_eff = C/dt
        В статике: не добавляем ничего (разрыв цепи)
        """
        if self.dt is None:
            return  # Статический анализ - конденсатор не вносит вклад
        
        G = self.G_eff
        G_matrix = G * np.array([
            [ 1, -1],
            [-1,  1]
        ])
        
        idx1 = index_map[self.node1][0]
        idx2 = index_map[self.node2][0]
        global_indices = [idx1, idx2]
        
        for i, gi in enumerate(global_indices):
            for j, gj in enumerate(global_indices):
                A[gi, gj] += G_matrix[i, j]
    
    def contribute_to_b(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Вклад от предыдущего состояния: G_eff * V_old
        """
        if self.dt is None:
            return  # Статический анализ
        
        idx1 = index_map[self.node1][0]
        idx2 = index_map[self.node2][0]
        
        I_history = self.G_eff * self.V_old
        
        b[idx1] += I_history
        b[idx2] -= I_history
    
    def get_current(self, V1_new: float, V2_new: float) -> float:
        """
        Вычислить ток через конденсатор
        
        Returns:
            Ток I [А]
        """
        if self.dt is None:
            return 0.0  # DC анализ - нет тока
        
        V_new = V1_new - V2_new
        I = self.C * (V_new - self.V_old) / self.dt
        return I
    
    def update_state(self, V1_new: float, V2_new: float):
        """
        Обновить состояние конденсатора после шага по времени
        """
        self.V_old = V1_new - V2_new


class Inductor(Contribution):
    """
    Катушка индуктивности - реактивный элемент (накапливает энергию в магнитном поле).
    
    Соотношение: V = L * dI/dt
    где I - ток через катушку
    
    Для статического анализа (DC) катушка = короткое замыкание (V = 0).
    Для динамического анализа используется дискретизация.
    
    При использовании неявной схемы Эйлера:
    dI/dt ≈ (I_new - I_old) / dt
    
    Получаем: V = L * (I_new - I_old) / dt
    Или: I_new = (V * dt / L) + I_old
    
    Эффективное сопротивление: R_eff = L / dt
    Эффективная проводимость: G_eff = dt / L
    """
    
    def __init__(self,
                 node1: Variable,
                 node2: Variable,
                 L: float,           # индуктивность [Гн]
                 dt: float = None,   # шаг по времени [с]
                 I_old: float = 0.0): # ток на предыдущем шаге
        """
        Args:
            node1: Переменная потенциала первого узла
            node2: Переменная потенциала второго узла
            L: Индуктивность [Гн]
            dt: Шаг по времени для динамического анализа [с]
            I_old: Ток через катушку на предыдущем шаге [А]
        """
        if node1.size != 1 or node2.size != 1:
            raise ValueError("Узлы должны быть скалярами")
        
        if L <= 0:
            raise ValueError("Индуктивность должна быть положительной")
        
        self.node1 = node1
        self.node2 = node2
        self.L = L
        self.dt = dt
        self.I_old = I_old
        
        # Для динамического анализа
        if dt is not None:
            if dt <= 0:
                raise ValueError("Шаг по времени должен быть положительным")
            self.G_eff = dt / L  # эффективная проводимость
        else:
            self.G_eff = float('inf')  # В статике катушка = короткое замыкание
    
    def get_variables(self) -> List[Variable]:
        return [self.node1, self.node2]
    
    def contribute_to_A(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        В динамике: G_eff = dt/L
        В статике: бесконечная проводимость (короткое замыкание)
        """
        if self.dt is None:
            # Статический анализ - идеальный проводник
            # Устанавливаем очень большую проводимость
            G = 1e10
        else:
            G = self.G_eff
        
        G_matrix = G * np.array([
            [ 1, -1],
            [-1,  1]
        ])
        
        idx1 = index_map[self.node1][0]
        idx2 = index_map[self.node2][0]
        global_indices = [idx1, idx2]
        
        for i, gi in enumerate(global_indices):
            for j, gj in enumerate(global_indices):
                A[gi, gj] += G_matrix[i, j]
    
    def contribute_to_b(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Вклад от предыдущего тока
        """
        if self.dt is None:
            return  # Статический анализ
        
        idx1 = index_map[self.node1][0]
        idx2 = index_map[self.node2][0]
        
        # Эквивалентный источник тока от истории
        I_history = self.I_old
        
        b[idx1] -= I_history
        b[idx2] += I_history
    
    def get_current(self, V1: float, V2: float) -> float:
        """
        Вычислить ток через катушку
        
        Returns:
            Ток I [А]
        """
        if self.dt is None:
            # DC анализ: ток определяется из решения системы
            V = V1 - V2
            # В DC должно быть V ≈ 0 (короткое замыкание)
            return 0.0  # Нужно получать из решения
        
        V = V1 - V2
        I_new = self.G_eff * V + self.I_old
        return I_new
    
    def update_state(self, V1: float, V2: float):
        """
        Обновить состояние катушки после шага по времени
        """
        if self.dt is not None:
            self.I_old = self.get_current(V1, V2)


class VoltageSource(Contribution):
    """
    Идеальный источник напряжения.
    
    Задает фиксированную разность потенциалов между узлами:
    V1 - V2 = V_source
    
    Реализуется через метод множителей Лагранжа или большого числа.
    Здесь используем метод "большого числа": добавляем очень большую
    проводимость и соответствующий источник тока.
    """
    
    def __init__(self,
                 node1: Variable,  # положительный полюс
                 node2: Variable,  # отрицательный полюс (может быть земля)
                 V: float):        # напряжение [В]
        """
        Args:
            node1: Переменная потенциала положительного узла
            node2: Переменная потенциала отрицательного узла
            V: Напряжение источника [В] (V1 - V2 = V)
        """
        if node1.size != 1 or node2.size != 1:
            raise ValueError("Узлы должны быть скалярами")
        
        self.node1 = node1
        self.node2 = node2
        self.V = V
        
        # Большое число для численной реализации ограничения
        self.G_big = 1e10
    
    def get_variables(self) -> List[Variable]:
        return [self.node1, self.node2]
    
    def contribute_to_A(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Добавляем очень большую проводимость
        """
        G = self.G_big
        G_matrix = G * np.array([
            [ 1, -1],
            [-1,  1]
        ])
        
        idx1 = index_map[self.node1][0]
        idx2 = index_map[self.node2][0]
        global_indices = [idx1, idx2]
        
        for i, gi in enumerate(global_indices):
            for j, gj in enumerate(global_indices):
                A[gi, gj] += G_matrix[i, j]
    
    def contribute_to_b(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Добавляем эквивалентный источник тока: I = G_big * V
        """
        idx1 = index_map[self.node1][0]
        idx2 = index_map[self.node2][0]
        
        I_eq = self.G_big * self.V
        
        b[idx1] += I_eq
        b[idx2] -= I_eq
    
    def get_current(self, V1: float, V2: float) -> float:
        """
        Вычислить ток через источник напряжения
        
        Примечание: точный ток нужно вычислять из баланса токов в узле,
        здесь возвращаем оценку на основе отклонения от заданного напряжения.
        """
        V_actual = V1 - V2
        return self.G_big * (self.V - V_actual)


class CurrentSource(Contribution):
    """
    Идеальный источник тока.
    
    Задает фиксированный ток, втекающий в node1 и вытекающий из node2.
    Это просто добавляет константу в правую часть системы уравнений.
    """
    
    def __init__(self,
                 node1: Variable,  # узел, куда втекает ток
                 node2: Variable,  # узел, откуда вытекает ток
                 I: float):        # ток [А]
        """
        Args:
            node1: Переменная потенциала узла, в который втекает ток
            node2: Переменная потенциала узла, из которого вытекает ток
            I: Ток источника [А] (положительный = от node2 к node1)
        """
        if node1.size != 1 or node2.size != 1:
            raise ValueError("Узлы должны быть скалярами")
        
        self.node1 = node1
        self.node2 = node2
        self.I = I
    
    def get_variables(self) -> List[Variable]:
        return [self.node1, self.node2]
    
    def contribute_to_A(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Источник тока не влияет на матрицу системы
        """
        pass
    
    def contribute_to_b(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Добавляем ток в правую часть
        """
        idx1 = index_map[self.node1][0]
        idx2 = index_map[self.node2][0]
        
        # Ток втекает в node1 и вытекает из node2
        b[idx1] += self.I
        b[idx2] -= self.I
    
    def get_voltage(self, V1: float, V2: float) -> float:
        """
        Получить напряжение на источнике тока
        
        Returns:
            Напряжение V [В]
        """
        return V1 - V2
    
    def get_power(self, V1: float, V2: float) -> float:
        """
        Вычислить мощность источника
        
        Returns:
            Мощность P [Вт] (положительная = отдает энергию)
        """
        V = self.get_voltage(V1, V2)
        return V * self.I


class Ground(Contribution):
    """
    Заземление - фиксирует потенциал узла в ноль.
    
    Это граничное условие, аналогичное закреплению в механике.
    """
    
    def __init__(self, node: Variable):
        """
        Args:
            node: Переменная потенциала узла, который заземляется
        """
        if node.size != 1:
            raise ValueError("Узел должен быть скаляром")
        
        self.node = node
        self.G_big = 1e10  # Большое число для реализации ограничения
    
    def get_variables(self) -> List[Variable]:
        return [self.node]
    
    def contribute_to_A(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Добавляем большое число на диагональ
        """
        idx = index_map[self.node][0]
        A[idx, idx] += self.G_big
    
    def contribute_to_b(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Правая часть = 0 (потенциал = 0)
        """
        # Если добавить G_big * 0 = 0, то ничего не меняется
        pass
