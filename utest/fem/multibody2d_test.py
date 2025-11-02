#!/usr/bin/env python3
# coding:utf-8

import unittest
import numpy as np
import warnings
from termin.fem.assembler import MatrixAssembler
from termin.fem.multibody2d import (
    RotationalInertia2D,
    TorqueSource2D,
    RigidBody2D,
    ForceVector2D
)


class TestIntegrationMultibody2D(unittest.TestCase):
    """Интеграционные тесты для многотельной системы"""
    
    def test_free_rotation_with_torque(self):
        """
        Тест свободного вращения с постоянным моментом.
        Без демпфирования ω должна расти линейно: ω = τ*t/J
        """
        assembler = MatrixAssembler()
        omega = assembler.add_variable("omega", size=1)
        
        J = 2.0  # кг·м²
        torque = 10.0  # Н·м
        dt = 0.01  # с
        
        inertia = RotationalInertia2D(omega, J=J, B=0.0, dt=dt, omega_old=0.0)
        torque_source = TorqueSource2D(omega, torque=torque)
        
        assembler.add_contribution(inertia)
        assembler.add_contribution(torque_source)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assembler.solve_and_set()
        
        # Ожидаемое ускорение: α = τ/J
        # За один шаг: ω_new = ω_old + α*dt = 0 + (10/2)*0.01 = 0.05
        # Но для неявной схемы: (J/dt)*ω_new = τ + (J/dt)*ω_old
        # ω_new = τ*dt/J = 10*0.01/2 = 0.05
        expected_omega = torque * dt / J
        
        self.assertAlmostEqual(omega.value, expected_omega, places=5)
    
    def test_damped_rotation(self):
        """
        Тест вращения с демпфированием без внешнего момента.
        Скорость должна затухать.
        """
        assembler = MatrixAssembler()
        omega = assembler.add_variable("omega", size=1)
        
        J = 1.0
        B = 10.0
        dt = 0.01
        omega_old = 10.0  # начальная скорость
        
        inertia = RotationalInertia2D(omega, J=J, B=B, dt=dt, omega_old=omega_old)
        assembler.add_contribution(inertia)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assembler.solve_and_set()
        
        # (J/dt + B)*ω_new = (J/dt)*ω_old
        # ω_new = ω_old * (J/dt) / (J/dt + B)
        expected_omega = omega_old * (J / dt) / (J / dt + B)
        
        self.assertAlmostEqual(omega.value, expected_omega, places=5)
        
        # Новая скорость должна быть меньше начальной
        self.assertLess(omega.value, omega_old)
    
    def test_rigid_body_with_force(self):
        """
        Тест твердого тела с приложенной силой.
        """
        assembler = MatrixAssembler()
        velocity = assembler.add_variable("velocity", size=2)
        omega = assembler.add_variable("omega", size=1)
        
        m = 2.0  # кг
        J = 0.5  # кг·м²
        dt = 0.01  # с
        force = np.array([20.0, 0.0])  # Н
        
        body = RigidBody2D(
            velocity, omega, m=m, J=J, C=0.0, B=0.0, dt=dt,
            v_old=np.zeros(2), omega_old=0.0
        )
        force_element = ForceVector2D(velocity, force)
        
        assembler.add_contribution(body)
        assembler.add_contribution(force_element)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assembler.solve_and_set()
        
        # v_new = F*dt/m = 20*0.01/2 = 0.1
        expected_vx = force[0] * dt / m
        expected_vy = 0.0
        
        self.assertAlmostEqual(velocity.value[0], expected_vx, places=5)
        self.assertAlmostEqual(velocity.value[1], expected_vy, places=5)
        self.assertAlmostEqual(omega.value, 0.0, places=5)

