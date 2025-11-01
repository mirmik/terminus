import unittest
from termin.kinematics import Transform3
from termin.geombase import Pose3
import numpy
import math

class TestTransform3(unittest.TestCase):
    def test_relocate_and_global_pose(self):
        transform = Transform3()

        pose = Pose3(
            ang=numpy.array([0.0, 0.0, math.sin(math.pi/4), math.cos(math.pi/4)]),
            lin=numpy.array([1.0, 2.0, 3.0])
        )
        transform.relocate(pose)
        global_pose = transform.global_pose()
        numpy.testing.assert_array_almost_equal(global_pose.ang, pose.ang)
        numpy.testing.assert_array_almost_equal(global_pose.lin, pose.lin)


    def test_hierarchy_global_pose(self):
        parent = Transform3()
        child = Transform3(parent=parent)

        parent_pose = Pose3(
            ang=numpy.array([0.0, 0.0, math.sin(math.pi/4), math.cos(math.pi/4)]),
            lin=numpy.array([1.0, 0.0, 0.0])
        )
        child_pose = Pose3(
            ang=numpy.array([0.0, 0.0, math.sin(math.pi/4), math.cos(math.pi/4)]),
            lin=numpy.array([0.0, 1.0, 0.0])
        )

        parent.relocate(parent_pose)
        child.relocate(child_pose)

        expected_global_child_pose = parent_pose * child_pose
        global_child_pose = child.global_pose()

        numpy.testing.assert_array_almost_equal(global_child_pose.ang, expected_global_child_pose.ang)
        numpy.testing.assert_array_almost_equal(global_child_pose.lin, expected_global_child_pose.lin)

    def test_transform_point(self):
        transform = Transform3()
        pose = Pose3.moveX(1.0) * Pose3.rotateZ(math.pi/2)
        transform.relocate(pose)

        point = numpy.array([1.0, 0.0, 0.0])
        transformed_point = transform.transform_point(point)
        expected_point = numpy.array([1.0, 1.0, 0.0])  # After 90 deg rotation around Z and translation

        numpy.testing.assert_array_almost_equal(transformed_point, expected_point)