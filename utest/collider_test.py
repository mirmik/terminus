from termin.colliders.capsule import CapsuleCollider
from termin.colliders.sphere import SphereCollider
from termin.colliders.union_collider import UnionCollider
import unittest
import numpy

class TestCollider(unittest.TestCase):
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

    def test_closest_union_collider(self):
        sphere1 = SphereCollider(
            center = numpy.array([0.0, 0.0, 0.0]),
            radius = 0.5
        )
        sphere2 = SphereCollider(
            center = numpy.array([3.0, 0.0, 0.0]),
            radius = 0.5
        )
        union_collider = UnionCollider([sphere1, sphere2])

        test_sphere = SphereCollider(
            center = numpy.array([0.0, 1.5, 0.0]),
            radius = 0.5
        )

        p_near, q_near, dist = union_collider.closest_to_collider(test_sphere)

        expected_dist = 0.5  # Closest to sphere1
        self.assertAlmostEqual(dist, expected_dist)

        expected_p_near = numpy.array([0.0, 0.5, 0.0])
        expected_q_near = numpy.array([0.0, 1.0, 0.0])

        numpy.testing.assert_array_almost_equal(p_near, expected_p_near)
        numpy.testing.assert_array_almost_equal(q_near, expected_q_near)

    def test_two_union_colliders(self):
        sphere1 = SphereCollider(
            center = numpy.array([0.0, 0.0, 0.0]),
            radius = 0.5
        )
        sphere2 = SphereCollider(
            center = numpy.array([3.0, 0.0, 0.0]),
            radius = 0.5
        )
        union_collider1 = UnionCollider([sphere1, sphere2])

        sphere3 = SphereCollider(
            center = numpy.array([1.25, 0.0, 0.0]),
            radius = 0.5
        )
        sphere4 = SphereCollider(
            center = numpy.array([5.0, 0.0, 0.0]),
            radius = 0.5
        )
        union_collider2 = UnionCollider([sphere3, sphere4])

        p_near, q_near, dist = union_collider1.closest_to_collider(union_collider2)

        expected_dist = 0.25  # Closest between sphere2 and sphere3
        self.assertAlmostEqual(dist, expected_dist)

        expected_p_near = numpy.array([0.5, 0.0, 0.0])
        expected_q_near = numpy.array([0.75, 0.0, 0.0])

        numpy.testing.assert_array_almost_equal(p_near, expected_p_near)
        numpy.testing.assert_array_almost_equal(q_near, expected_q_near)
        