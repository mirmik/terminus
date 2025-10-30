import unittest
from termin.ga201.motor import Motor2
from termin.ga201.screw import Screw2
import math
import numpy


def early(a, b):
    if abs(a.x - b.x) > 0.0001:
        return False
    if abs(a.y - b.y) > 0.0001:
        return False
    if abs(a.z - b.z) > 0.0001:
        return False
    return True

def screw_equal(a, b):
    if abs(a.moment() - b.moment()) > 0.0001:
        return False
    if abs(a.vector()[0] - b.vector()[0]) > 0.0001:
        return False
    if abs(a.vector()[1] - b.vector()[1]) > 0.0001:
        return False
    return True

class TransformationProbe(unittest.TestCase):
    def test_moment_carry(self):
        motor = Motor2.translation(1, 0)
        screw = Screw2(m=1, v=[0,0])
        carried = screw.kinematic_carry(motor)
        invcarried = carried.kinematic_carry(motor.inverse())
        invcarried2 = carried.inverse_kinematic_carry(motor)
        self.assertEqual(carried.moment(), 1)
        self.assertTrue((carried.vector() == numpy.array([0,-1])).all())
        self.assertTrue(screw_equal(invcarried, screw))
        self.assertTrue(screw_equal(invcarried2, screw))

    def test_vectory_carry(self):
        motor = Motor2.translation(1, 0)
        screw = Screw2(m=0, v=[0,1])
        carried = screw.kinematic_carry(motor)
        self.assertEqual(carried.moment(), 0)
        self.assertTrue((carried.vector() == numpy.array([0,1])).all())

    def test_vectorx_carry(self):
        motor = Motor2.translation(1, 0)
        screw = Screw2(m=0, v=[1,0])
        carried = screw.kinematic_carry(motor)
        self.assertEqual(carried.moment(), 0)
        self.assertTrue((carried.vector() == numpy.array([1,0])).all())

    def test_moment_carry_with_rotation(self):
        motor = Motor2.translation(1, 0) * Motor2.rotation(math.pi/2)
        screw = Screw2(m=1, v=[0,0])
        carried = screw.kinematic_carry(motor)
        invcarried = carried.kinematic_carry(motor.inverse())
        invcarried2 = carried.inverse_kinematic_carry(motor)
        self.assertEqual(carried.moment(), 1)
        self.assertTrue((carried.vector() == numpy.array([0,-1])).all())
        self.assertTrue(screw_equal(invcarried, screw))
        self.assertTrue(screw_equal(invcarried2, screw))

    def test_vectory_carry_with_rotation(self):
        motor = Motor2.translation(1, 0) * Motor2.rotation(math.pi/2)
        screw = Screw2(m=0, v=[0,1])
        carried = screw.kinematic_carry(motor)
        self.assertEqual(carried.moment(), 0)
        self.assertTrue(screw_equal(carried, Screw2(m=0, v=[-1,0])))

    def test_vectorx_carry_with_rotation(self):
        motor = Motor2.translation(1, 0) * Motor2.rotation(math.pi/2)
        screw = Screw2(m=0, v=[1,0])
        carried = screw.kinematic_carry(motor)
        self.assertEqual(carried.moment(), 0)
        self.assertTrue(screw_equal(carried, Screw2(m=0, v=[0,1])))