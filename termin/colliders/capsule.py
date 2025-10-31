
from termin.clossest import closest_points_between_segments, closest_points_between_capsules, closest_points_between_capsule_and_sphere
import numpy
from termin.colliders.collider import Collider
from termin.colliders.sphere import SphereCollider

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

    def closest_to_union_collider(self, other: "UnionCollider"):
        """Return the closest points and distance between this capsule and a union collider."""
        a,b,c = other.closest_to_collider(self)
        return b,a,c

    def closest_to_collider(self, other: "Collider"):
        """Return the closest points and distance between this collider and another collider."""

        from .capsule import SphereCollider
        from .union_collider import UnionCollider

        if isinstance(other, CapsuleCollider):
            return self.closest_to_capsule(other)
        elif isinstance(other, SphereCollider):
            return other.closest_to_sphere(self)
        elif isinstance(other, UnionCollider):
            return self.closest_to_union_collider(other)
        else:
            raise NotImplementedError(f"closest_to_collider not implemented for {type(other)}")

    def distance(self, other: "Collider") -> float:
        """Return the distance between this collider and another collider."""
        _, _, dist = self.closest_to_collider(other)
        return dist
