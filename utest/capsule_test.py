from terminus.colliders.capsule import CapsuleCollider, SphereCollider
import unittest
import numpy

class TestCapsuleCollider(unittest.TestCase):
    def test_closest_to_capsule(self):
        capsule1 = CapsuleCollider(
            a = numpy.array([0.0, 0.0, 0.0]),
            b = numpy.array([0.0, 0.0, 1.0]),
            radius = 0.25
        )
        capsule2 = CapsuleCollider(
            a = numpy.array([1.0, 0.0, 0.0]),
            b = numpy.array([1.0, 0.0, 0.0]),
            radius = 0.25
        )

        p_near, q_near, dist = capsule1.closest_to_capsule(capsule2)

        expected_dist = 0.5  # Capsules are touching
        self.assertAlmostEqual(dist, expected_dist)

        expected_p_near = numpy.array([0.25, 0.0, 0.0])
        expected_q_near = numpy.array([0.75, 0.0, 0.0])

        numpy.testing.assert_array_almost_equal(p_near, expected_p_near)
        numpy.testing.assert_array_almost_equal(q_near, expected_q_near)

    def test_closest_sphere_to_capsule(self):
        capsule = CapsuleCollider(
            a = numpy.array([0.0, 0.0, 0.0]),
            b = numpy.array([0.0, 0.0, 2.0]),
            radius = 0.5
        )
        sphere_center = numpy.array([1.0, 0.0, 1.0])
        sphere_radius = 0.5

        sphere = SphereCollider(
            center = sphere_center,
            radius = sphere_radius
        )

        p_near, q_near, dist = sphere.closest_to_capsule(capsule)

        expected_dist = 0.0  # They are touching
        self.assertAlmostEqual(dist, expected_dist)

        expected_p_near = numpy.array([0.5, 0.0, 1.0])
        expected_q_near = numpy.array([0.5, 0.0, 1.0])

        numpy.testing.assert_array_almost_equal(p_near, expected_p_near)
        numpy.testing.assert_array_almost_equal(q_near, expected_q_near)