#!/usr/bin/env python3
# coding:utf-8

import unittest
import numpy as np
import warnings
from termin.fem.dynamic_assembler import DynamicMatrixAssembler
from termin.fem.multibody3d_2 import (
    RigidBody3D, #ForceOnBody3D, FixedRotationJoint3D, RevoluteJoint3D
)
from numpy import linalg
from termin.fem.inertia3d import SpatialInertia3D


class TestIntegrationMultibody3D(unittest.TestCase):
    """Интеграционные тесты для многотельной системы"""

    def test_rigid_body_with_gravity(self):
        """Создание простой системы с одним твердым телом и гравитацией"""
        assembler = DynamicMatrixAssembler()
        
        body = RigidBody3D(
            inertia=SpatialInertia3D(mass=2.0, inertia=np.diag([1.0, 1.0, 1.0]), com=np.zeros(3)),
            gravity=np.array([0.0, 0.0, -9.81]),
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
        assert np.isclose(x[2], -9.81)
