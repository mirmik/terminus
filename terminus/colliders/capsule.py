
from terminus.clossest import closest_points_between_segments, closest_points_between_capsules, closest_points_between_capsule_and_sphere
import numpy
from terminus.colliders.collider import Collider

class SphereCollider(Collider):
    def __init__(self, center: numpy.ndarray, radius: float):
        self.center = center
        self.radius = radius

    def transform_by(self, transform: 'Transform3'):
        """Return a new SphereCollider transformed by the given Transform3."""
        new_center = transform.transform_point(self.center)
        return SphereCollider(new_center, self.radius)

    def closest_to_sphere(self, other: "SphereCollider"):
        """Return the closest points and distance between this sphere and another sphere."""
        center_dist = numpy.linalg.norm(other.center - self.center)
        dist = center_dist - (self.radius + other.radius)
        if center_dist > 1e-8:
            dir_vec = (other.center - self.center) / center_dist
        else:
            dir_vec = numpy.array([1.0, 0.0, 0.0])  # Arbitrary direction if centers coincide
        p_near = self.center + dir_vec * self.radius
        q_near = other.center - dir_vec * other.radius
        return p_near, q_near, dist

    def closest_to_capsule(self, other: "CapsuleCollider"):
        """Return the closest points and distance between this sphere and a capsule."""
        p_near, q_near, dist = closest_points_between_capsule_and_sphere(
            other.a, other.b, other.radius,
            self.center, self.radius)
        return q_near, p_near, dist

    def closest_to_collider(self, other: "Collider"):
        """Return the closest points and distance between this collider and another collider."""
        if isinstance(other, SphereCollider):
            return self.closest_to_sphere(other)
        elif isinstance(other, CapsuleCollider):
            return self.closest_to_capsule(other)
        else:
            raise NotImplementedError("closest_to_collider not implemented for this collider type.")

class CapsuleCollider(Collider):
    def __init__(self, a: numpy.ndarray, b: numpy.ndarray, radius: float):
        self.a = a
        self.b = b
        self.radius = radius

    def transform_by(self, transform: 'Transform3'):
        """Return a new CapsuleCollider transformed by the given Transform3."""
        new_a = transform.transform_point(self.a)
        new_b = transform.transform_point(self.b)
        return CapsuleCollider(new_a, new_b, self.radius)
    
    def closest_to_capsule(self, other: "CapsuleCollider"):
        """Return the closest points and distance between this capsule and another capsule."""
        p_near, q_near, dist = closest_points_between_capsules(
            self.a, self.b, self.radius,
            other.a, other.b, other.radius)
        return p_near, q_near, dist

    def closest_to_sphere(self, other: "SphereCollider"):
        """Return the closest points and distance between this capsule and a sphere."""
        p_near, q_near, dist = closest_points_between_capsule_and_sphere(
            self.a, self.b, self.radius,
            other.center, other.radius)
        return p_near, q_near, dist

    def closest_to_collider(self, other: "Collider"):
        """Return the closest points and distance between this collider and another collider."""
        if isinstance(other, CapsuleCollider):
            return self.closest_to_capsule(other)
        elif isinstance(other, SphereCollider):
            return other.closest_to_sphere(self)
        else:
            raise NotImplementedError("closest_to_collider not implemented for this collider type.")