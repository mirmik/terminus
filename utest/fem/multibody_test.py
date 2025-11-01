#!/usr/bin/env python3
"""
Тесты для многотельной механики (fem/multibody.py)
"""

import unittest
import numpy as np
import sys
import os

# Добавить путь к модулю
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from termin.fem.multibody import (
    RotationalInertia, TorqueSource,
    RotationalSpring, RotationalDamper, FixedRotation
)
from termin.fem.assembler import Variable, MatrixAssembler


def solve_system(contributions, variables):
    """
    Вспомогательная функция для решения системы
    """
    assembler = MatrixAssembler()
    assembler.variables = variables
    
    for contrib in contributions:
        assembler.contributions.append(contrib)
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assembler.solve_and_set()


class TestRotationalInertia(unittest.TestCase):
    """Тесты для вращательной инерции"""
    
    def test_inertia_with_torque(self):
        """
        Инерция с приложенным моментом (статика)
        
        τ = 10 Н·м, B = 2 Н·м·с
        
        В установившемся режиме: B*ω = τ
        ω = τ/B = 10/2 = 5 рад/с
        """
        omega = Variable("omega", 1)
        
        J = 0.1  # кг·м²
        B = 2.0  # Н·м·с
        
        inertia = RotationalInertia(omega, J, B)
        torque_source = TorqueSource(omega, 10.0)
        
        solve_system(
            [inertia, torque_source],
            [omega]
        )
        
        expected_omega = 10.0 / B
        self.assertAlmostEqual(omega.value, expected_omega, places=4)
    
    def test_inertia_dynamics(self):
        """
        Динамика инерции: разгон под действием момента
        
        J*dω/dt = τ - B*ω
        
        При постоянном моменте система стремится к ω_final = τ/B
        """
        omega = Variable("omega", 1)
        
        J = 0.1  # кг·м²
        B = 2.0  # Н·м·с
        tau = 10.0  # Н·м
        dt = 0.01  # с
        
        inertia = RotationalInertia(omega, J, B, dt=dt, omega_old=0.0)
        torque_source = TorqueSource(omega, tau)
        
        omega_values = [0.0]
        
        # Симуляция разгона
        for step in range(50):
            solve_system(
                [inertia, torque_source],
                [omega]
            )
            
            omega_values.append(omega.value)
            inertia.update_state(omega.value)
        
        # Проверить, что скорость растет
        self.assertGreater(omega_values[10], omega_values[1])
        self.assertGreater(omega_values[30], omega_values[10])
        
        # Финальная скорость должна стремиться к tau/B
        omega_final = tau / B
        self.assertGreater(omega_values[-1], 0.7 * omega_final)
        self.assertLess(omega_values[-1], omega_final * 1.1)


class TestRotationalSpring(unittest.TestCase):
    """Тесты для вращательной пружины"""
    
    def test_spring_coupling(self):
        """
        Две инерции, связанные пружиной
        
        τ -> [J1, B1] --[пружина]-- [J2, B2]
        
        Момент передается через пружину
        """
        omega1 = Variable("omega1", 1)
        omega2 = Variable("omega2", 1)
        
        J1 = 0.1
        J2 = 0.2
        B1 = 1.0
        B2 = 1.0
        K_spring = 10.0
        tau = 5.0
        
        inertia1 = RotationalInertia(omega1, J1, B1)
        inertia2 = RotationalInertia(omega2, J2, B2)
        spring = RotationalSpring(omega1, omega2, K_spring)
        torque = TorqueSource(omega1, tau)
        
        solve_system(
            [inertia1, inertia2, spring, torque],
            [omega1, omega2]
        )
        
        # Обе скорости должны быть положительными
        self.assertGreater(omega1.value, 0.0)
        self.assertGreater(omega2.value, 0.0)
        
        # omega1 должно быть больше omega2 (пружина под нагрузкой)
        self.assertGreater(omega1.value, omega2.value)


class TestFixedRotation(unittest.TestCase):
    """Тесты для фиксированной скорости"""
    
    def test_fixed_rotation(self):
        """
        Фиксация угловой скорости
        """
        omega = Variable("omega", 1)
        
        fixed = FixedRotation(omega, 10.0)
        
        solve_system([fixed], [omega])
        
        self.assertAlmostEqual(omega.value, 10.0, places=4)


if __name__ == '__main__':
    unittest.main()
