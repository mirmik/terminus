import unittest
from termin.geombase import Pose3
from termin.util import deg2rad
import numpy
import math

class TestPose3(unittest.TestCase):
    def test_identity(self):
        pose = Pose3.identity()
        point = numpy.array([1.0, 2.0, 3.0])
        transformed_point = pose.transform_point(point)
        numpy.testing.assert_array_almost_equal(transformed_point, point)

    def test_inverse(self):
        pose = Pose3(
            ang=numpy.array([0.0, 0.0, math.sin(math.pi/4), math.cos(math.pi/4)]),
            lin=numpy.array([1.0, 2.0, 3.0])
        )
        inv_pose = pose.inverse()
        point = numpy.array([4.0, 5.0, 6.0])
        transformed_point = pose.transform_point(point)
        recovered_point = inv_pose.transform_point(transformed_point)
        numpy.testing.assert_array_almost_equal(recovered_point, point)

    def test_inverse2(self):
        pose = Pose3(
            ang=numpy.array([0.0, 0.0, math.sin(math.pi/4), math.cos(math.pi/4)]),
            lin=numpy.array([1.0, 2.0, 3.0])
        )
        inv_pose = pose.inverse()
    
        m = pose * inv_pose
        numpy.testing.assert_array_almost_equal(m.lin, [0,0,0])
        numpy.testing.assert_array_almost_equal(m.ang, [0,0,0,1])

        m = inv_pose*pose
        numpy.testing.assert_array_almost_equal(m.lin, [0,0,0])
        numpy.testing.assert_array_almost_equal(m.ang, [0,0,0,1])


    def test_inverse_3(self):
        pose = Pose3.translation(1,0,0) * Pose3.rotation(
            numpy.array([1,0,0]), deg2rad(10))
        inv_pose = pose.inverse()
        point = numpy.array([4.0, 5.0, 6.0])
        transformed_point = pose.transform_point(point)
        recovered_point = inv_pose.transform_point(transformed_point)
        numpy.testing.assert_array_almost_equal(recovered_point, point)




    def test_rotation(self):
        angle = math.pi / 2  # 90 degrees
        pose = Pose3(
            ang=numpy.array([0.0, 0.0, math.sin(angle/2), math.cos(angle/2)]),
            lin=numpy.array([0.0, 0.0, 0.0])
        )
        point = numpy.array([1.0, 0.0, 0.0])
        transformed_point = pose.transform_point(point)
        expected_point = numpy.array([0.0, 1.0, 0.0])
        numpy.testing.assert_array_almost_equal(transformed_point, expected_point)

    def test_rotation_x(self):
        angle = math.pi / 2  # 90 degrees
        pose = Pose3(
            ang=numpy.array([math.sin(angle/2), 0.0, 0.0, math.cos(angle/2)]),
            lin=numpy.array([0.0, 0.0, 0.0])
        )
        point = numpy.array([0.0, 0.0, 1.0])
        transformed_point = pose.transform_point(point)
        expected_point = numpy.array([0.0, -1.0, 0.0])
        numpy.testing.assert_array_almost_equal(transformed_point, expected_point)

    def test_translation(self):
        pose = Pose3(
            ang=numpy.array([0.0, 0.0, 0.0, 1.0]),
            lin=numpy.array([1.0, 2.0, 3.0])
        )
        point = numpy.array([4.0, 5.0, 6.0])
        transformed_point = pose.transform_point(point)
        expected_point = numpy.array([5.0, 7.0, 9.0])
        numpy.testing.assert_array_almost_equal(transformed_point, expected_point)

    def test_composition(self):
        pose1 = Pose3(
            ang=numpy.array([0.0, 0.0, math.sin(math.pi/4), math.cos(math.pi/4)]),
            lin=numpy.array([1.0, 0.0, 0.0])
        )
        pose2 = Pose3(
            ang=numpy.array([0.0, 0.0, math.sin(math.pi/4), math.cos(math.pi/4)]),
            lin=numpy.array([0.0, 1.0, 0.0])
        )
        composed_pose = pose1 * pose2
        point = numpy.array([1.0, 0.0, 0.0])
        transformed_point = composed_pose.transform_point(point)

        # Manually compute expected result
        intermediate_point = pose2.transform_point(point)
        expected_point = pose1.transform_point(intermediate_point)

        numpy.testing.assert_array_almost_equal(transformed_point, expected_point)

    def test_lerp(self):
        pose1 = Pose3(
            ang=numpy.array([0.0, 0.0, math.sin(0.0), math.cos(0.0)]),
            lin=numpy.array([0.0, 0.0, 0.0])
        )
        pose2 = Pose3(
            ang=numpy.array([0.0, 0.0, math.sin(math.pi/2), math.cos(math.pi/2)]),
            lin=numpy.array([10.0, 0.0, 0.0])
        )
        t = 0.5
        lerped_pose = Pose3.lerp(pose1, pose2, t)

        point = numpy.array([1.0, 0.0, 0.0])
        transformed_point = lerped_pose.transform_point(point)

        # Expected rotation is 45 degrees around Z and translation is (5, 0, 0)
        expected_rotation = Pose3(
            ang=numpy.array([0.0, 0.0, math.sin(math.pi/4), math.cos(math.pi/4)]),
            lin=numpy.array([5.0, 0.0, 0.0])
        )
        expected_point = expected_rotation.transform_point(point)

        numpy.testing.assert_array_almost_equal(transformed_point, expected_point)