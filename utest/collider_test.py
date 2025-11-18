from termin.colliders.capsule import CapsuleCollider
from termin.colliders.sphere import SphereCollider
from termin.colliders.union_collider import UnionCollider
from termin.colliders.box import BoxCollider
import unittest
from termin.kinematic import Transform3
from termin.colliders.attached import AttachedCollider
import numpy
from termin.geombase import AABB

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

    def test_closest_of_box_and_capsule(self):
        box = BoxCollider(
            center = numpy.array([0.0, 0.0, 0.0]),
            size = numpy.array([2.0, 1.0, 0.5])
        )
        capsule = CapsuleCollider(
            a = numpy.array([3.0, 0.0, 0.0]),
            b = numpy.array([4.0, 0.0, 0.0]),
            radius = 0.2
        )

        closest_box_point, closest_capsule_point, distance = box.closest_point_to_capsule(capsule)

        expected_distance = 1.8
        self.assertAlmostEqual(distance, expected_distance)

        expected_closest_box_point = numpy.array([1.0, 0.0, 0.0])
        expected_closest_capsule_point = numpy.array([2.8, 0.0, 0.0])

        numpy.testing.assert_array_almost_equal(closest_box_point, expected_closest_box_point)
        numpy.testing.assert_array_almost_equal(closest_capsule_point, expected_closest_capsule_point)

class TestColliderAvoidance(unittest.TestCase):
    def test_avoidance_vector(self):
        sphere1 = SphereCollider(
            center = numpy.array([0.0, 0.0, 0.0]),
            radius = 0.5
        )
        sphere2 = SphereCollider(
            center = numpy.array([6.0, 0.0, 0.0]),
            radius = 0.5
        )

        direction, dist, closest_point = sphere1.avoidance(sphere2)

        expected_direction = numpy.array([-1.0, 0.0, 0.0])
        expected_dist = 5.0  # Distance between surfaces
        expected_closest_point = numpy.array([0.5, 0.0, 0.0])

        numpy.testing.assert_array_almost_equal(direction, expected_direction)
        numpy.testing.assert_array_almost_equal(closest_point, expected_closest_point)
        self.assertAlmostEqual(dist, expected_dist)

class AttachedColliderTest(unittest.TestCase):
    def test_attached_collider_distance(self):
        box = BoxCollider(
            center = numpy.array([0.0, 0.0, 0.0]),
            size = numpy.array([2.0, 1.0, 0.5])
        )
        numpy.testing.assert_array_almost_equal(box.local_aabb().min_point, numpy.array([-1.0, -0.5, -0.25]))
        numpy.testing.assert_array_almost_equal(box.local_aabb().max_point, numpy.array([1.0, 0.5, 0.25]))

        trans = Transform3()
        attached_box = AttachedCollider(box, trans)

        sphere = SphereCollider(
            center = numpy.array([3.0, 0.0, 0.0]),
            radius = 0.5
        )
        attached_sphere = AttachedCollider(sphere, Transform3())
        p_near, q_near, dist = attached_box.closest_to_collider(attached_sphere)

        expected_distance = 1.5  # Distance between surfaces
        expected_p_near = numpy.array([1.0, 0.0, 0.0])
        expected_q_near = numpy.array([2.5, 0.0, 0.0])

        self.assertAlmostEqual(dist, expected_distance)
        numpy.testing.assert_array_almost_equal(p_near, expected_p_near)
        numpy.testing.assert_array_almost_equal(q_near, expected_q_near)
