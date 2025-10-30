from termin.aabb import AABB
import unittest
import numpy

class AABBTest(unittest.TestCase):
    def test_aabb_creation(self):
        points = numpy.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [-1.0, -2.0, -3.0]
        ])
        aabb = AABB.from_points(points)
        expected_min = numpy.array([-1.0, -2.0, -3.0])
        expected_max = numpy.array([4.0, 5.0, 6.0])

        numpy.testing.assert_array_equal(aabb.min_point, expected_min)
        numpy.testing.assert_array_equal(aabb.max_point, expected_max)

    def test_aabb_merge(self):
        aabb1 = AABB(numpy.array([0.0, 0.0, 0.0]), numpy.array([1.0, 1.0, 1.0]))
        aabb2 = AABB(numpy.array([0.5, 0.5, 0.5]), numpy.array([2.0, 2.0, 2.0]))

        merged_aabb = aabb1.merge(aabb2)
        expected_min = numpy.array([0.0, 0.0, 0.0])
        expected_max = numpy.array([2.0, 2.0, 2.0])

        numpy.testing.assert_array_equal(merged_aabb.min_point, expected_min)
        numpy.testing.assert_array_equal(merged_aabb.max_point, expected_max)

    def test_intersects(self):
        aabb1 = AABB(numpy.array([0.0, 0.0]), numpy.array([1.0, 1.0]))
        aabb2 = AABB(numpy.array([0.5, 0.5]), numpy.array([1.5, 1.5]))
        aabb3 = AABB(numpy.array([2.0, 2.0]), numpy.array([3.0, 3.0]))

        self.assertTrue(aabb1.intersects(aabb2))
        self.assertFalse(aabb1.intersects(aabb3))

        aabb4 = AABB(numpy.array([-0.5, 0.5]), numpy.array([0.5, 1.5]))
        self.assertTrue(aabb1.intersects(aabb4))