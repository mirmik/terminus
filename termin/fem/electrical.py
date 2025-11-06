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
                 R: float,         # сопротивление [Ом]
                 assembler=None):  # ассемблер для автоматической регистрации
        """
        Args:
            node1: Переменная потенциала первого узла (скаляр)
            node2: Переменная потенциала второго узла (скаляр)
            R: Сопротивление [Ом]
            assembler: MatrixAssembler для автоматической регистрации переменных
        """
        if node1.size != 1 or node2.size != 1:
            raise ValueError("Узлы должны быть скалярами (потенциалы)")
        
        if R <= 0:
            raise ValueError("Сопротивление должно быть положительным")
        
        super().__init__([node1, node2], assembler)
        
        self.node1 = node1
        self.node2 = node2
        self.R = R
        self.G = 1.0 / R  # проводимость
        self.G_matrix = np.array([
            [ 1, -1],
            [-1,  1]
        ]) * self.G
    
    def contribute_to_damping(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Добавляет вклад резистора в матрицу проводимости
        """
        G = self.G
        
        idx1 = index_map[self.node1][0]
        idx2 = index_map[self.node2][0]
        global_indices = [idx1, idx2]
        
        for i, gi in enumerate(global_indices):
            for j, gj in enumerate(global_indices):
                A[gi, gj] += self.G_matrix[i, j]


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
                 assembler=None):    # ассемблер для автоматической регистрации
        """
        Args:
            node1: Переменная потенциала первого узла
            node2: Переменная потенциала второго узла
            C: Емкость [Ф]
            dt: Шаг по времени для динамического анализа [с] (None для статики)
            V_old: Напряжение на конденсаторе на предыдущем шаге [В]
            assembler: MatrixAssembler для автоматической регистрации переменных
        """
        if node1.size != 1 or node2.size != 1:
            raise ValueError("Узлы должны быть скалярами")
        
        if C <= 0:
            raise ValueError("Емкость должна быть положительной")
        
        super().__init__([node1, node2], assembler)
        
        self.node1 = node1
        self.node2 = node2
        self.C = C
        self.C_matrix = np.array([
            [ 1, -1],
            [-1,  1]
        ]) * C
        
    def contribute_to_stiffness(self, M: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Добавляет вклад конденсатора в матрицу массы
        """
        C = self.C
        
        idx1 = index_map[self.node1][0]
        idx2 = index_map[self.node2][0]
        global_indices = [idx1, idx2]
        
        for i, gi in enumerate(global_indices):
            for j, gj in enumerate(global_indices):
                M[gi, gj] += self.C_matrix[i, j]


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
                 assembler=None):    # ассемблер для автоматической регистрации
        """
        Args:
            node1: Переменная потенциала первого узла
            node2: Переменная потенциала второго узла
            L: Индуктивность [Гн]
            dt: Шаг по времени для динамического анализа [с]
            I_old: Ток через катушку на предыдущем шаге [А]
            assembler: MatrixAssembler для автоматической регистрации переменных
        """
        if node1.size != 1 or node2.size != 1:
            raise ValueError("Узлы должны быть скалярами")
        
        if L <= 0:
            raise ValueError("Индуктивность должна быть положительной")
        
        super().__init__([node1, node2], assembler)
        
        self.node1 = node1
        self.node2 = node2
        self.L = L
        self.L_matrix = np.array([
            [ 1, -1],
            [-1,  1]
        ]) * L

    def contribute_to_mass(self, M: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Добавляет вклад катушки в матрицу массы
        """
        L = self.L
        
        idx1 = index_map[self.node1][0]
        idx2 = index_map[self.node2][0]
        global_indices = [idx1, idx2]
        
        for i, gi in enumerate(global_indices):
            for j, gj in enumerate(global_indices):
                M[gi, gj] += self.L_matrix[i, j]



class VoltageSource(Constrait):
    """
    Идеальный источник напряжения.
    
    Задает фиксированную разность потенциалов между узлами:
    V1 - V2 = V_source
    """
    
    def __init__(self,
                 node1: Variable,  # положительный полюс
                 node2: Variable,  # отрицательный полюс (может быть земля)
                 V: float,         # напряжение [В]
                 assembler=None):  # ассемблер для автоматической регистрации
        """
        Args:
            node1: Переменная потенциала положительного узла
            node2: Переменная потенциала отрицательного узла
            V: Напряжение источника [В] (V1 - V2 = V)
            assembler: MatrixAssembler для автоматической регистрации переменных
        """
        if node1.size != 1 or node2.size != 1:
            raise ValueError("Узлы должны быть скалярами")
        
        self.current = 

        super().__init__([node1, node2], assembler)
        
        self.node1 = node1
        self.node2 = node2
        self.V = V

    def contribute_to_holonomic_load(self, b: np.ndarray, holonomic_index_map: Dict[Variable, List[int]]):
        """
        Добавляем ограничение на разность потенциалов
        """
        idx1 = holonomic_index_map[self.node1][0]
        idx2 = holonomic_index_map[self.node2][0]
        
        # Ограничение: V1 - V2 = V
        b[idx1] += self.V
        b[idx2] -= self.V

        
class CurrentSource(Contribution):
    """
    Идеальный источник тока.
    
    Задает фиксированный ток, втекающий в node1 и вытекающий из node2.
    Это просто добавляет константу в правую часть системы уравнений.
    """
    
    def __init__(self,
                 node1: Variable,  # узел, куда втекает ток
                 node2: Variable,  # узел, откуда вытекает ток
                 I: float,         # ток [А]
                 assembler=None):  # ассемблер для автоматической регистрации
        """
        Args:
            node1: Переменная потенциала узла, в который втекает ток
            node2: Переменная потенциала узла, из которого вытекает ток
            I: Ток источника [А] (положительный = от node2 к node1)
            assembler: MatrixAssembler для автоматической регистрации переменных
        """
        if node1.size != 1 or node2.size != 1:
            raise ValueError("Узлы должны быть скалярами")
        
        super().__init__([node1, node2], assembler)
        
        self.node1 = node1
        self.node2 = node2
        self.I = I
    
    def contribute_to_load(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
        """
        Добавляем вклад источника тока в правую часть
        """
        idx1 = index_map[self.node1][0]
        idx2 = index_map[self.node2][0]
        
        b[idx1] += self.I   # ток втекает в node1
        b[idx2] -= self.I   # ток вытекает из node2


# class Ground(Contribution):
#     """
#     Заземление - фиксирует потенциал узла в ноль.
    
#     Это граничное условие, аналогичное закреплению в механике.
#     """
    
#     def __init__(self, node: Variable, assembler=None):
#         """
#         Args:
#             node: Переменная потенциала узла, который заземляется
#             assembler: MatrixAssembler для автоматической регистрации переменных
#         """
#         if node.size != 1:
#             raise ValueError("Узел должен быть скаляром")
        
#         super().__init__([node], assembler)
        
#         self.node = node
#         self.G_big = 1e10  # Большое число для реализации ограничения
    
#     def contribute_to_mass(self, A: np.ndarray, index_map: Dict[Variable, List[int]]):
#         """
#         Добавляем большое число на диагональ
#         """
#         idx = index_map[self.node][0]
#         A[idx, idx] += self.G_big
    
#     def contribute_to_b(self, b: np.ndarray, index_map: Dict[Variable, List[int]]):
#         """
#         Правая часть = 0 (потенциал = 0)
#         """
#         # Если добавить G_big * 0 = 0, то ничего не меняется
#         pass
