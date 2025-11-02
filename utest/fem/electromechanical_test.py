#!/usr/bin/env python3
"""
Тесты для электромеханических элементов (fem/electromechanical.py)

Содержит тесты только для DCMotor - класса, связывающего электрическую
и механическую подсистемы.
"""

import unittest
import numpy as np
import sys
import os

# Добавить путь к модулю
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from termin.fem.electromechanical import DCMotor
from termin.fem.multibody2d import RotationalInertia2D
from termin.fem.electrical import VoltageSource, Ground
from termin.fem.assembler import Variable, MatrixAssembler, LagrangeConstraint


def fixed_scalar(variable: Variable, value: float = 0.0):
    """
    Создать constraint для фиксации скалярной переменной
    
    Args:
        variable: Переменная размера 1
        value: Целевое значение
    
    Returns:
        LagrangeConstraint для фиксации переменной
    """
    return LagrangeConstraint(
        variables=[variable],
        coefficients=[np.array([[1.0]])],  # просто x = value
        rhs=np.array([value])
    )


def solve_system(contributions, variables, constraints=None):
    """
    Вспомогательная функция для решения системы
    
    Args:
        contributions: Список Contribution объектов
        variables: Список Variable объектов
        constraints: Список Constraint объектов (опционально)
    """
    assembler = MatrixAssembler()
    assembler.variables = variables
    
    for contrib in contributions:
        assembler.contributions.append(contrib)
    
    if constraints:
        assembler.constraints = constraints
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        use_constraints = constraints is not None and len(constraints) > 0
        assembler.solve_and_set(use_constraints=use_constraints)


class TestDCMotor(unittest.TestCase):
    """Тесты для двигателя постоянного тока"""
    
    def test_motor_static_no_load(self):
        """
        Двигатель без нагрузки в статике
        
        10V -> Двигатель (R=1Ω, K_e=0.1 В/(рад/с)) -> GND
        ω фиксирована = 0
        
        Ожидаем: I = V/R = 10А, τ = K_t*I
        """
        v_plus = Variable("V+", 1)
        v_gnd = Variable("GND", 1)
        omega = Variable("omega", 1)
        
        R = 1.0
        L = 0.01
        K_e = 0.1
        K_t = 0.1
        
        v_source = VoltageSource(v_plus, v_gnd, 10.0)
        motor = DCMotor(v_plus, v_gnd, omega, R, L, K_e, K_t)
        ground = Ground(v_gnd)
        fixed_rotation = fixed_scalar(omega, 0.0)  # зафиксировать ω=0
        
        solve_system(
            [v_source, motor, ground],
            [v_plus, v_gnd, omega],
            constraints=[fixed_rotation]
        )
        
        # Проверить напряжение
        self.assertAlmostEqual(v_plus.value, 10.0, places=4)
        self.assertAlmostEqual(omega.value, 0.0, places=4)
        
        # Проверить ток
        I = motor.get_current(v_plus.value, v_gnd.value, omega.value)
        self.assertAlmostEqual(I, 10.0, places=4)
        
        # Проверить момент
        torque = motor.get_torque(I=I)
        self.assertAlmostEqual(torque, K_t * 10.0, places=4)
    
    def test_motor_with_back_emf(self):
        """
        Двигатель с ЭДС противодействия
        
        10V -> Двигатель -> GND
        ω = 50 рад/с (зафиксирована)
        K_e = 0.1 В/(рад/с)
        
        ЭДС = K_e*ω = 0.1*50 = 5В
        I = (V - EMF)/R = (10 - 5)/1 = 5А
        """
        v_plus = Variable("V+", 1)
        v_gnd = Variable("GND", 1)
        omega = Variable("omega", 1)
        
        R = 1.0
        L = 0.01
        K_e = 0.1
        K_t = 0.1
        
        v_source = VoltageSource(v_plus, v_gnd, 10.0)
        motor = DCMotor(v_plus, v_gnd, omega, R, L, K_e, K_t)
        ground = Ground(v_gnd)
        fixed_rotation = fixed_scalar(omega, 50.0)
        
        solve_system(
            [v_source, motor, ground],
            [v_plus, v_gnd, omega],
            constraints=[fixed_rotation]
        )
        
        self.assertAlmostEqual(omega.value, 50.0, places=4)
        
        I = motor.get_current(v_plus.value, v_gnd.value, omega.value)
        expected_I = (10.0 - K_e * 50.0) / R
        self.assertAlmostEqual(I, expected_I, places=4)


class TestMotorWithLoad(unittest.TestCase):
    """Тесты для двигателя с механической нагрузкой"""
    
    def test_motor_steady_state(self):
        """
        Двигатель с инерционной нагрузкой в установившемся режиме
        
        Электрическое уравнение: V = R*I + K_e*ω
        Механическое уравнение: K_t*I = B*ω (в установившемся режиме)
        
        Решая систему:
        I = K_t*I*B/K_e  =>  ω = V/(K_e + R*B/K_t)
        """
        v_plus = Variable("V+", 1)
        v_gnd = Variable("GND", 1)
        omega = Variable("omega", 1)
        
        V = 12.0
        R = 1.0
        L = 0.01
        K_e = 0.1
        K_t = 0.1
        J = 0.01
        B = 0.05
        
        v_source = VoltageSource(v_plus, v_gnd, V)
        ground = Ground(v_gnd)
        motor = DCMotor(v_plus, v_gnd, omega, R, L, K_e, K_t)
        inertia = RotationalInertia2D(omega, J, B)
        
        solve_system(
            [v_source, ground, motor, inertia],
            [v_plus, v_gnd, omega]
        )
        
        # Проверить, что получили разумные значения
        self.assertGreater(omega.value, 0.0)
        self.assertLess(omega.value, 200.0)
        
        # Проверить баланс: момент двигателя = момент трения
        I = motor.get_current(v_plus.value, v_gnd.value, omega.value)
        motor_torque = motor.get_torque(I=I)
        friction_torque = B * omega.value
        
        self.assertAlmostEqual(motor_torque, friction_torque, places=2)
    
    def test_motor_acceleration(self):
        """
        Разгон двигателя с инерционной нагрузкой
        """
        v_plus = Variable("V+", 1)
        v_gnd = Variable("GND", 1)
        omega = Variable("omega", 1)
        
        V = 12.0
        R = 1.0
        L = 0.01
        K_e = 0.1
        K_t = 0.1
        J = 0.01
        B = 0.05
        dt = 0.001
        
        v_source = VoltageSource(v_plus, v_gnd, V)
        ground = Ground(v_gnd)
        motor = DCMotor(v_plus, v_gnd, omega, R, L, K_e, K_t, dt=dt, I_old=0.0)
        inertia = RotationalInertia2D(omega, J, B, dt=dt)
        
        omega_values = [0.0]
        I_values = [0.0]
        
        # Симуляция разгона
        for step in range(100):
            solve_system(
                [v_source, ground, motor, inertia],
                [v_plus, v_gnd, omega]
            )
            
            omega_values.append(omega.value)
            
            I = motor.get_current(v_plus.value, v_gnd.value, omega.value)
            I_values.append(I)
        
        # Проверить, что скорость растет
        self.assertGreater(omega_values[30], omega_values[10])
        self.assertGreater(omega_values[70], omega_values[30])
        
        # Проверить, что система стабилизируется
        # (скорость не растет бесконечно)
        self.assertLess(omega_values[-1], 200.0)


if __name__ == '__main__':
    unittest.main()
