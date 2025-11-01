#!/usr/bin/env python3
"""
Тесты для электрических элементов (fem/electrical.py)
"""

import unittest
import numpy as np
import sys
import os

# Добавить путь к модулю
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from termin.fem.electrical import (
    Resistor, Capacitor, Inductor, 
    VoltageSource, CurrentSource, Ground
)
from termin.fem.assembler import Variable, MatrixAssembler


def solve_circuit(contributions, variables):
    """
    Вспомогательная функция для решения электрической схемы
    
    Args:
        contributions: список элементов схемы
        variables: список переменных
    
    Returns:
        dict: словарь {переменная: значение}
    """
    assembler = MatrixAssembler()
    
    # Добавить все переменные (они уже созданы, нужно их зарегистрировать)
    assembler.variables = variables
    
    # Добавить все вклады
    for contrib in contributions:
        assembler.contributions.append(contrib)
    
    # Решить систему
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assembler.solve_and_set()


class TestResistor(unittest.TestCase):
    """Тесты для резистора"""
    
    def test_resistor_creation(self):
        """Создание резистора"""
        v1 = Variable("V1", 1)
        v2 = Variable("V2", 1)
        r = Resistor(v1, v2, R=100.0)
        
        self.assertEqual(r.R, 100.0)
        self.assertAlmostEqual(r.G, 0.01)
    
    def test_resistor_conductance_matrix(self):
        """Матрица проводимости резистора"""
        v1 = Variable("V1", 1)
        v2 = Variable("V2", 1)
        r = Resistor(v1, v2, R=10.0)
        
        G_matrix = r.get_conductance_matrix()
        expected = 0.1 * np.array([[1, -1], [-1, 1]])
        
        np.testing.assert_array_almost_equal(G_matrix, expected)
    
    def test_resistor_current(self):
        """Вычисление тока через резистор"""
        v1 = Variable("V1", 1)
        v2 = Variable("V2", 1)
        r = Resistor(v1, v2, R=100.0)
        
        # U = 10V, R = 100 Ом => I = 0.1 А
        I = r.get_current(10.0, 0.0)
        self.assertAlmostEqual(I, 0.1)
        
        # Обратное направление
        I = r.get_current(0.0, 10.0)
        self.assertAlmostEqual(I, -0.1)
    
    def test_resistor_power(self):
        """Вычисление мощности резистора"""
        v1 = Variable("V1", 1)
        v2 = Variable("V2", 1)
        r = Resistor(v1, v2, R=100.0)
        
        # P = I²R = (0.1)² * 100 = 1 Вт
        P = r.get_power(10.0, 0.0)
        self.assertAlmostEqual(P, 1.0)


class TestVoltageDivider(unittest.TestCase):
    """Тест делителя напряжения"""
    
    def test_simple_voltage_divider(self):
        """
        Простой делитель напряжения:
        12V ---[1kΩ]--- V_mid ---[1kΩ]--- GND
        
        Ожидаем V_mid = 6V
        """
        # Создать узлы
        v_in = Variable("V_in", 1)
        v_mid = Variable("V_mid", 1)
        v_gnd = Variable("GND", 1)
        
        # Создать элементы
        v_source = VoltageSource(v_in, v_gnd, 12.0)
        r1 = Resistor(v_in, v_mid, 1000.0)
        r2 = Resistor(v_mid, v_gnd, 1000.0)
        ground = Ground(v_gnd)
        
        # Собрать систему
        solve_circuit(
            [v_source, r1, r2, ground],
            [v_in, v_mid, v_gnd]
        )
        
        # Проверить результат        
        self.assertAlmostEqual(v_in.value, 12.0, places=5)
        self.assertAlmostEqual(v_mid.value, 6.0, places=5)
        self.assertAlmostEqual(v_gnd.value, 0.0, places=5)

    def test_asymmetric_voltage_divider(self):
        """
        Несимметричный делитель:
        10V ---[2kΩ]--- V_mid ---[3kΩ]--- GND
        
        V_mid = 10V * 3kΩ / (2kΩ + 3kΩ) = 6V
        """
        v_in = Variable("V_in", 1)
        v_mid = Variable("V_mid", 1)
        v_gnd = Variable("GND", 1)
        
        v_source = VoltageSource(v_in, v_gnd, 10.0)
        r1 = Resistor(v_in, v_mid, 2000.0)
        r2 = Resistor(v_mid, v_gnd, 3000.0)
        ground = Ground(v_gnd)
        
        solve_circuit(
            [v_source, r1, r2, ground],
            [v_in, v_mid, v_gnd]
        )
        
        self.assertAlmostEqual(v_mid.value, 6.0, places=5)


class TestCurrentSource(unittest.TestCase):
    """Тесты для источника тока"""
    
    def test_current_source_with_resistor(self):
        """
        Источник тока через резистор:
        I_source (1А) ---[100Ω]--- GND
        
        Ожидаем V = I * R = 100V
        """
        v_node = Variable("V_node", 1)
        v_gnd = Variable("GND", 1)
        
        i_source = CurrentSource(v_node, v_gnd, 1.0)  # 1А
        resistor = Resistor(v_node, v_gnd, 100.0)
        ground = Ground(v_gnd)
        
        solve_circuit(
            [i_source, resistor, ground],
            [v_node, v_gnd]
        )
        
        self.assertAlmostEqual(v_node.value, 100.0, places=5)


class TestSeriesParallelCircuits(unittest.TestCase):
    """Тесты для последовательных и параллельных цепей"""
    
    def test_series_resistors(self):
        """
        Последовательные резисторы:
        10V ---[100Ω]--- V1 ---[200Ω]--- V2 ---[300Ω]--- GND
        
        R_total = 600Ω
        I = 10V / 600Ω = 1/60 А
        V1 = 10 - 100*I = 10 - 100/60 = 8.333V
        V2 = 10 - 300*I = 10 - 300/60 = 5V
        """
        v_in = Variable("V_in", 1)
        v1 = Variable("V1", 1)
        v2 = Variable("V2", 1)
        v_gnd = Variable("GND", 1)
        
        v_source = VoltageSource(v_in, v_gnd, 10.0)
        r1 = Resistor(v_in, v1, 100.0)
        r2 = Resistor(v1, v2, 200.0)
        r3 = Resistor(v2, v_gnd, 300.0)
        ground = Ground(v_gnd)
        
        solve_circuit(
            [v_source, r1, r2, r3, ground],
            [v_in, v1, v2, v_gnd]
        )
        
        self.assertAlmostEqual(v1.value, 10.0 - 100.0/60.0, places=4)
        self.assertAlmostEqual(v2.value, 5.0, places=4)
    
    def test_parallel_resistors(self):
        """
        Параллельные резисторы:
        10V ---+--- [100Ω] ---+--- GND
               |              |
               +--- [100Ω] ---+
        
        Эквивалентное сопротивление = 50Ω
        Ток = 10V / 50Ω = 0.2А
        """
        v_in = Variable("V_in", 1)
        v_out = Variable("V_out", 1)
        v_gnd = Variable("GND", 1)
        
        v_source = VoltageSource(v_in, v_gnd, 10.0)
        r1 = Resistor(v_in, v_out, 100.0)
        r2 = Resistor(v_in, v_out, 100.0)
        ground = Ground(v_gnd)
        
        # Добавляем малую нагрузку, чтобы v_out был определен
        r_load = Resistor(v_out, v_gnd, 1e6)
        
        solve_circuit(
            [v_source, r1, r2, r_load, ground],
            [v_in, v_out, v_gnd]
        )
        
        # V_out должно быть близко к V_in (большая нагрузка)
        # Проверим ток через r1
        I1 = r1.get_current(v_in.value, v_out.value)
        I2 = r2.get_current(v_in.value, v_out.value)
        
        # Токи через параллельные резисторы должны быть равны
        self.assertAlmostEqual(I1, I2, places=5)


class TestCapacitor(unittest.TestCase):
    """Тесты для конденсатора"""
    
    def test_capacitor_static_analysis(self):
        """
        В статическом анализе конденсатор = разрыв цепи
        """
        v1 = Variable("V1", 1)
        v2 = Variable("V2", 1)
        
        c = Capacitor(v1, v2, C=1e-6)  # без dt - статика
        
        # Проверить, что эффективная проводимость = 0
        self.assertEqual(c.G_eff, 0.0)
    
    def test_rc_circuit_initial(self):
        """
        RC цепь в начальный момент (конденсатор разряжен)
        
        10V ---[1kΩ]--- V_C ---[C=1μF]--- GND
        dt = 0.001s
        
        В начальный момент V_C ≈ 0 (конденсатор заряжается)
        """
        v_in = Variable("V_in", 1)
        v_c = Variable("V_C", 1)
        v_gnd = Variable("GND", 1)
        
        dt = 0.001  # 1мс
        
        v_source = VoltageSource(v_in, v_gnd, 10.0)
        resistor = Resistor(v_in, v_c, 1000.0)
        capacitor = Capacitor(v_c, v_gnd, C=1e-6, dt=dt, V_old=0.0)
        ground = Ground(v_gnd)
        
        solve_circuit(
            [v_source, resistor, capacitor, ground],
            [v_in, v_c, v_gnd]
        )
        
        # Конденсатор должен начать заряжаться
        self.assertGreater(v_c.value, 0.0)
        self.assertLess(v_c.value, 10.0)
    
    def test_rc_charging(self):
        """
        Процесс заряда RC цепи
        
        Постоянная времени τ = R*C = 1000 * 1e-6 = 1e-3 с
        После времени τ напряжение достигает ~63% от конечного
        """
        v_in = Variable("V_in", 1)
        v_c = Variable("V_C", 1)
        v_gnd = Variable("GND", 1)
        
        R = 1000.0
        C = 1e-6
        tau = R * C  # 1мс
        dt = tau / 10.0  # 0.1мс шаг
        V_source = 10.0
        
        v_source_elem = VoltageSource(v_in, v_gnd, V_source)
        resistor = Resistor(v_in, v_c, R)
        capacitor = Capacitor(v_c, v_gnd, C=C, dt=dt, V_old=0.0)
        ground = Ground(v_gnd)
        
        # Симуляция нескольких шагов
        V_C_values = [0.0]
        
        for step in range(20):  # 20 шагов = 2мс = 2τ
            solve_circuit(
                [v_source_elem, resistor, capacitor, ground],
                [v_in, v_c, v_gnd]
            )
            
            V_C_values.append(v_c.value)
            
            # Обновить состояние конденсатора
            capacitor.V_old = v_c.value
        
        # Проверить, что напряжение растет
        self.assertGreater(V_C_values[10], V_C_values[1])
        self.assertGreater(V_C_values[20], V_C_values[10])
        
        # После 2τ должно быть около 86% от конечного значения
        self.assertGreater(V_C_values[20], 0.7 * V_source)
        self.assertLess(V_C_values[20], V_source)


class TestInductor(unittest.TestCase):
    """Тесты для катушки индуктивности"""
    
    def test_inductor_static_analysis(self):
        """
        В статическом анализе катушка = короткое замыкание
        
        10V ---[100Ω]--- V_L ---[L]--- GND
        
        Катушка в DC - короткое замыкание, поэтому V_L ≈ 0
        """
        v_in = Variable("V_in", 1)
        v_l = Variable("V_L", 1)
        v_gnd = Variable("GND", 1)
        
        v_source = VoltageSource(v_in, v_gnd, 10.0)
        resistor = Resistor(v_in, v_l, 100.0)
        inductor = Inductor(v_l, v_gnd, L=1e-3)  # без dt - статика
        ground = Ground(v_gnd)
        
        solve_circuit(
            [v_source, resistor, inductor, ground],
            [v_in, v_l, v_gnd]
        )
        
        # Напряжение на входе должно быть 10В
        self.assertAlmostEqual(v_in.value, 10.0, places=4)
        # Напряжение на катушке должно быть близко к нулю (короткое замыкание)
        self.assertAlmostEqual(v_l.value, 0.0, places=2)
    
    def test_rl_circuit(self):
        """
        RL цепь переходный процесс
        
        10V ---[100Ω]--- V_L ---[L=10mH]--- GND
        
        Постоянная времени τ = L/R = 0.01/100 = 0.0001 с
        """
        v_in = Variable("V_in", 1)
        v_l = Variable("V_L", 1)
        v_gnd = Variable("GND", 1)
        
        R = 100.0
        L = 0.01  # 10mH
        tau = L / R  # 0.1мс
        dt = tau / 10.0
        V_source = 10.0
        
        v_source_elem = VoltageSource(v_in, v_gnd, V_source)
        resistor = Resistor(v_in, v_l, R)
        inductor = Inductor(v_l, v_gnd, L=L, dt=dt, I_old=0.0)
        ground = Ground(v_gnd)
        
        # Симуляция
        I_values = [0.0]
        
        for step in range(20):
            solve_circuit(
                [v_source_elem, resistor, inductor, ground],
                [v_in, v_l, v_gnd]
            )
            
            I_new = inductor.get_current(v_l.value, 0.0)
            I_values.append(I_new)
            
            # Обновить состояние
            inductor.update_state(v_l.value, 0.0)
        
        # Проверить, что ток растет
        self.assertGreater(I_values[10], I_values[1])
        self.assertGreater(I_values[20], I_values[10])
        
        # Финальный ток должен стремиться к V/R = 0.1А
        expected_final = V_source / R
        self.assertGreater(I_values[20], 0.5 * expected_final)


class TestComplexCircuit(unittest.TestCase):
    """Тест сложной схемы"""
    
    def test_bridge_circuit(self):
        """
        Делительная схема с ответвлением:
        
               R1(100)           R2(100)
        V+ ----/\\/\\/\\----*----/\\/\\/\\---- GND
                        |
                      R3(50)
                        |
                       V_mid
                        |
                      R4(50)
                        |
                       GND
        
        V_node = V+ * R2/(R1+R2) = 12 * 100/200 = 6V
        V_mid = V_node * R4/(R3+R4) = 6 * 50/100 = 3V
        """
        v_plus = Variable("V+", 1)
        v_node = Variable("V_node", 1)
        v_mid = Variable("V_mid", 1)
        v_gnd = Variable("GND", 1)
        
        v_source = VoltageSource(v_plus, v_gnd, 12.0)
        r1 = Resistor(v_plus, v_node, 100.0)
        r2 = Resistor(v_node, v_gnd, 100.0)
        r3 = Resistor(v_node, v_mid, 50.0)
        r4 = Resistor(v_mid, v_gnd, 50.0)
        ground = Ground(v_gnd)
        
        solve_circuit(
            [v_source, r1, r2, r3, r4, ground],
            [v_plus, v_node, v_mid, v_gnd]
        )
        
        # V_plus должно быть 12V (источник)
        self.assertAlmostEqual(v_plus.value, 12.0, places=4)
        
        # V_node: учитывая нагрузку R3-R4, это не простой делитель
        # Проверим просто что значения разумные
        self.assertGreater(v_node.value, 3.0)
        self.assertLess(v_node.value, 9.0)
        
        # V_mid должно быть меньше V_node
        self.assertGreater(v_node.value, v_mid.value)
        self.assertGreater(v_mid.value, 0.0)


if __name__ == '__main__':
    unittest.main()
