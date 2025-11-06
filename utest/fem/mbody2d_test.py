#!/usr/bin/env python3
# coding:utf-8

import unittest
import numpy as np
import warnings
from termin.fem.assembler import DynamicMatrixAssembler
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

        joint = FixedRotationJoint2D(
            body=body,
            radius_to_body=np.array([1.0, 0.0]),
            assembler=assembler)

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
        assert np.isclose(x[2], 5.0)