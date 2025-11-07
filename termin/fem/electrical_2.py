import numpy as np
from typing import List, Dict
from .assembler import Contribution, Variable, Constraint   


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
    
    def contribute(self, matrices, index_maps: Dict[Variable, List[int]]):
        """
        Добавляет вклад резистора в матрицу проводимости
        """
        G = matrices["conductance"]
        index_map = index_maps["voltage"]  # предполагается, что переменные потен
        
        idx1 = index_map[self.node1][0]
        idx2 = index_map[self.node2][0]
        global_indices = [idx1, idx2]
        
        for i, gi in enumerate(global_indices):
            for j, gj in enumerate(global_indices):
                G[gi, gj] += self.G_matrix[i, j]

    def contribute_for_constraints_correction(self, matrices, index_maps):
        pass

class Capacitor(Contribution):
    """
    Идеальный конденсатор в DAE виде:
    q - C*(V1 - V2) = 0
    ток = dq/dt учитывается интегратором, а не элементом.
    """

    def __init__(self, node1: Variable, node2: Variable, C: float, assembler=None):
        if C <= 0:
            raise ValueError("Ёмкость должна быть > 0")

        self.node1 = node1
        self.node2 = node2
        self.C = C

        # заряд как переменная
        self.q = Variable("q_cap", size=1, tag="charge")

        super().__init__([node1, node2, self.q], assembler)

    def contribute(self, matrices, index_maps):
        H = matrices["holonomic"]            # уравнения ограничений
        poserr = matrices["position_error"]  # правая часть

        cmap = index_maps["holonomic_constraint_force"]
        vmap = index_maps["voltage"]
        qmap = index_maps["charge"]

        # одна строка ограничения
        row = cmap[self.q][0]

        v1 = vmap[self.node1][0]
        v2 = vmap[self.node2][0]
        q  = qmap[self.q][0]

        # q - C(V1 - V2) = 0
        H[row, q]  += 1.0
        H[row, v1] += -self.C
        H[row, v2] +=  self.C

        # ошибка положения (текущая)
        poserr[row] += self.q.value - self.C*(self.node1.value - self.node2.value)

    def contribute_for_constraints_correction(self, matrices, index_maps):
        self.contribute(matrices, index_maps)

class Inductor(Contribution):
    """
    Идеальная индуктивность:
    L * di/dt = V1 - V2
    В DAE виде: L*i_dot - (V1 - V2) = 0
    """

    def __init__(self, node1: Variable, node2: Variable, L: float, assembler=None):
        if L <= 0:
            raise ValueError("Индуктивность должна быть > 0")

        self.node1 = node1
        self.node2 = node2
        self.L = L

        # ток через индуктивность
        self.i = Variable("i_L", size=1, tag="current")

        super().__init__([node1, node2, self.i], assembler)

    def contribute(self, matrices, index_maps):
        H = matrices["holonomic"]
        velerr = matrices["holonomic_velocity_rhs"]   # правая часть для скоростей

        cmap = index_maps["holonomic_constraint_force"]
        vmap = index_maps["voltage"]
        imap = index_maps["current"]

        row = cmap[self.i][0]

        v1 = vmap[self.node1][0]
        v2 = vmap[self.node2][0]
        ii = imap[self.i][0]

        # L * i_dot - (V1 - V2) = 0 → в скоростях
        H[row, ii] += self.L
        H[row, v1] += -1.0
        H[row, v2] +=  1.0

        # правая часть: 0
        velerr[row] += 0.0

    def contribute_for_constraints_correction(self, matrices, index_maps):
        # Индуктивность не даёт ограничений на положение (только на скорости)
        pass

class VoltageSource(Contribution):
    """
    Идеальный источник напряжения: V1 - V2 = U
    """

    def __init__(self, node1, node2, U, assembler=None):
        self.node1 = node1
        self.node2 = node2
        self.U = float(U)

        # вводим ток как лагранжевый множитель
        self.i = Variable("i_vs", size=1, tag="current")

        super().__init__([node1, node2, self.i], assembler)

    def contribute(self, matrices, index_maps):
        H = matrices["holonomic"]
        poserr = matrices["position_error"]

        vmap = index_maps["voltage"]
        cmap = index_maps["holonomic_constraint_force"]

        row = cmap[self.i][0]
        v1 = vmap[self.node1][0]
        v2 = vmap[self.node2][0]

        # V1 - V2 = U
        H[row, v1] +=  1.0
        H[row, v2] += -1.0

        poserr[row] += (self.node1.value - self.node2.value - self.U)

    def contribute_for_constraints_correction(self, matrices, index_maps):
        self.contribute(matrices, index_maps)

class CurrentSource(Contribution):
    """
    Идеальный источник тока: +I в node1, -I в node2
    """
    def __init__(self, node1, node2, I, assembler=None):
        self.node1 = node1
        self.node2 = node2
        self.I = float(I)
        super().__init__([node1, node2], assembler)

    def contribute(self, matrices, index_maps):
        b = matrices["load"]
        vmap = index_maps["voltage"]

        i1 = vmap[self.node1][0]
        i2 = vmap[self.node2][0]

        b[i1] +=  self.I
        b[i2] += -self.I

    def contribute_for_constraints_correction(self, matrices, index_maps):
        pass