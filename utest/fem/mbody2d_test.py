#!/usr/bin/env python3
# coding:utf-8

import unittest
import numpy as np
import warnings
from termin.fem.dynamic_assembler import DynamicMatrixAssembler
from termin.fem.multibody2d_2 import (
    RigidBody2D, ForceOnBody2D, FixedRotationJoint2D
)
from numpy import linalg


class TestIntegrationMultibody2D(unittest.TestCase):
    """Интеграционные тесты для многотельной системы"""

    def test_rigid_body_with_gravity(self):
        """Создание простой системы с одним твердым телом и гравитацией"""
        assembler = DynamicMatrixAssembler()
        
        body = RigidBody2D(
            m=2.0,
            J=1.0,
            gravity=np.array([0.0, -9.81]),
            assembler=assembler)

        index_map = assembler.index_map()
        self.assertIn(body.velocity, index_map)

        matrices = assembler.assemble()

        self.assertIn("mass", matrices)
        self.assertIn("load", matrices)
        self.assertIn("stiffness", matrices)
        self.assertIn("damping", matrices)

        A_ext, b_ext = assembler.assemble_extended_system(matrices)
        x = linalg.solve(A_ext, b_ext)

        assert np.isclose(x[0], 0.0)
        assert np.isclose(x[1], -9.81)
        assert np.isclose(x[2], 0.0)

    def test_rigid_body_with_external_force(self):
        """Создание простой системы с одним твердым телом и внешней силой"""
        assembler = DynamicMatrixAssembler()
        
        body = RigidBody2D(
            m=3.0,
            J=2.0,
            gravity=np.array([0.0, 0.0]),
            assembler=assembler)

        force = ForceOnBody2D(
            body=body,
            force=np.array([6.0, 0.0]),
            torque=0.0,
            assembler=assembler)

        index_map = assembler.index_map()
        self.assertIn(body.velocity, index_map)

        matrices = assembler.assemble()

        self.assertIn("mass", matrices)
        self.assertIn("load", matrices)
        self.assertIn("stiffness", matrices)
        self.assertIn("damping", matrices)

        A_ext, b_ext = assembler.assemble_extended_system(matrices)
        x = linalg.solve(A_ext, b_ext)

        assert np.isclose(x[0], 2.0)  # vx = Fx/m = 6/3 = 2
        assert np.isclose(x[1], 0.0)
        assert np.isclose(x[2], 0.0) 

    def test_rigid_body_with_external_torque(self):
        """Создание простой системы с одним твердым телом и внешним моментом"""
        assembler = DynamicMatrixAssembler()
        
        body = RigidBody2D(
            m=4.0,
            J=8.0,
            gravity=np.array([0.0, 0.0]),
            assembler=assembler)

        force = ForceOnBody2D(
            body=body,
            force=np.array([0.0, 0.0]),
            torque=16.0,
            assembler=assembler)

        index_map = assembler.index_map()
        self.assertIn(body.velocity, index_map)

        matrices = assembler.assemble()

        self.assertIn("mass", matrices)
        self.assertIn("load", matrices)
        self.assertIn("stiffness", matrices)
        self.assertIn("damping", matrices)

        A_ext, b_ext = assembler.assemble_extended_system(matrices)
        x = linalg.solve(A_ext, b_ext)

        assert np.isclose(x[0], 0.0)  
        assert np.isclose(x[1], 0.0)
        assert np.isclose(x[2], 2.0)  # omega = torque/J = 16/8 = 2
    
    def test_rigid_body_with_fixed_rotation_joint(self):
        """Создание простой системы с одним твердым телом и фиксированным шарниром"""
        assembler = DynamicMatrixAssembler()
        
        body = RigidBody2D(
            m=5.0,
            J=5.0,
            gravity=np.array([0.0, -10.00]),
            assembler=assembler)

        body.velocity.set_value(np.array([1.0, 0.0]))
        body.omega.set_value(np.array([0.0]))

        joint = FixedRotationJoint2D(
            body=body,
            assembler=assembler)

        assert joint.radius_to_body is not None
        assert joint.radius_to_body[0] == 1.0
        assert joint.radius_to_body[1] == 0.0
        assert assembler.total_variables_by_tag("acceleration") == 3
        assert assembler.total_variables_by_tag("holonomic_constraint_force") == 2

        index_maps = assembler.index_maps()
        self.assertIn("acceleration", index_maps)
        self.assertIn("holonomic_constraint_force", index_maps)
        self.assertEqual(len(index_maps["holonomic_constraint_force"]), 1)
        self.assertEqual(len(index_maps["acceleration"]), 2)
        self.assertIn(body.velocity, index_maps["acceleration"])
        self.assertIn(joint.internal_force, index_maps["holonomic_constraint_force"])

        matrices = assembler.assemble()
        A_ext, b_ext = assembler.assemble_extended_system(matrices)
        x = linalg.solve(A_ext, b_ext)

        assert np.isclose(x[0], 0.0)
        assert np.isclose(x[1], -5.0)
        assert np.isclose(x[2], -5.0)

    def test_simple_pendulum_in_bottom_position(self):
        """Создание простой системы с одним твердым телом и фиксированным шарниром"""
        assembler = DynamicMatrixAssembler()
        
        body = RigidBody2D(
            m=5.0,
            J=5.0,
            gravity=np.array([0.0, -10.00]),
            assembler=assembler)

        body.velocity.set_value(np.array([0.0, -1.0]))
        body.omega.set_value(np.array([0.0]))

        joint = FixedRotationJoint2D(
            body=body,
            assembler=assembler)

        dt = 0.01  # временной шаг

        joint.update_radius_to_body()
        matrices = assembler.assemble()
        A_ext, b_ext = assembler.assemble_extended_system(matrices)
        x = linalg.solve(A_ext, b_ext)
        q_ddot, holonomic_lambdas, nonholonomic_lambdas = assembler.sort_results(x)
        q_dot = assembler.integrate_velocities(matrices["old_q_dot"], q_ddot, dt)
        q = assembler.integrate_positions(matrices["old_q"], q_dot, q_ddot, dt)
        assembler.upload_results(q_ddot, q_dot, q)
        assembler.integrate_nonlinear(dt)

        assert np.isclose(body.velocity.value_ddot[0], 0.0)
        assert np.isclose(body.velocity.value_ddot[1], 0.0)
        assert np.isclose(body.omega.value_ddot[0], 0.0)

    # def test_stabilization_test(self):
    #     """Создание простой системы с одним твердым телом и фиксированным шарниром"""
    #     assembler = DynamicMatrixAssembler()
        
    #     body = RigidBody2D(
    #         m=5.0,
    #         J=5.0,
    #         gravity=np.array([0.0, -10.00]),
    #         assembler=assembler)

    #     body.velocity.set_value(np.array([0.0, -1.0]))
    #     body.omega.set_value(np.array([0.0]))

    #     joint = FixedRotationJoint2D(
    #         body=body,
    #         assembler=assembler)

    #     dt = 0.01  # временной шаг
        
    #     body.velocity.set_value(np.array([0.01, -1.0]))
    #     body.omega.set_value(np.array([0.0]))

    #     for step in range(20):
    #         joint.update_radius_to_body()
    #         print(f"Step {step}: radius to body = {joint.radius_to_body}")

    #         matrices = assembler.assemble()
    #         A_ext, b_ext = assembler.assemble_extended_system(matrices)
    #         x = linalg.solve(A_ext, b_ext)
    #         q_ddot, holonomic_lambdas, nonholonomic_lambdas = assembler.sort_results(x)
    #         q_dot = assembler.integrate_velocities(matrices["old_q_dot"], q_ddot, dt)

    #         #print("holonomic:\n", matrices["holonomic"])
    #         #print("holonomic_load:\n", matrices["holonomic_load"])

    #         q_dot = assembler.restore_velocity_constraints(q_dot, matrices["holonomic"], matrices["holonomic_load"])
    #         #print("restored q_dot =", q_dot)

    #         #assert q_dot[0] != 0.0

    #         #print(matrices["damping"])

    #         q = assembler.integrate_positions(matrices["old_q"], q_dot, q_ddot, dt)
    #         #print("q =", q)
    #         #q = assembler.restore_position_constraints(q, matrices["holonomic"], matrices["holonomic_load"])
            
    #         #print("q after position restore =", q)

    #         #assert q[0] != 0.0

    #         assembler.upload_results(q_ddot, q_dot, q)
    #         assembler.integrate_nonlinear(dt)
    #         if step == 19:
    #             assert False

    # def test_simple_pendulum(self):
    #     """Создание простой системы с одним твердым телом и фиксированным шарниром"""
    #     assembler = DynamicMatrixAssembler()
        
    #     body = RigidBody2D(
    #         m=5.0,
    #         J=5.0,
    #         gravity=np.array([0.0, -10.00]),
    #         assembler=assembler)

    #     body.velocity.set_value(np.array([1.0, 0.0]))
    #     body.omega.set_value(np.array([0.0]))

    #     joint = FixedRotationJoint2D(
    #         body=body,
    #         assembler=assembler)

    #     dt = 0.01  # временной шаг

    #     for step in range(20):
    #         joint.update_radius_to_body()
    #         print(f"Step {step}: radius to body = {joint.radius_to_body}")

    #         matrices = assembler.assemble()
    #         A_ext, b_ext = assembler.assemble_extended_system(matrices)
    #         x = linalg.solve(A_ext, b_ext)
    #         q_ddot, holonomic_lambdas, nonholonomic_lambdas = assembler.sort_results(x)
    #         q_dot = assembler.integrate_velocities(matrices["old_q_dot"], q_ddot, dt)

    #         #print("holonomic:\n", matrices["holonomic"])
    #         #print("holonomic_load:\n", matrices["holonomic_load"])

    #         q_dot = assembler.restore_velocity_constraints(q_dot, matrices["holonomic"], matrices["holonomic_load"])
    #         #print("restored q_dot =", q_dot)

    #         #assert q_dot[0] != 0.0

    #         #print(matrices["damping"])

    #         q = assembler.integrate_positions(matrices["old_q"], q_dot, q_ddot, dt)
    #         #print("q =", q)
    #         #q = assembler.restore_position_constraints(q, matrices["holonomic"], matrices["holonomic_load"])
            
    #         #print("q after position restore =", q)

    #         #assert q[0] != 0.0

    #         assembler.upload_results(q_ddot, q_dot, q)
    #         assembler.integrate_nonlinear(dt)

            
            
    #         if step == 19:
    #             assert False