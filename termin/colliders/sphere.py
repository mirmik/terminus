
from termin.clossest import closest_points_between_segments, closest_points_between_capsules, closest_points_between_capsule_and_sphere
import numpy
from termin.colliders.collider import Collider
from termin.pose3 import Pose3



class SphereCollider(Collider):
    def __init__(self, center: numpy.ndarray, radius: float):
        self.center = center
        self.radius = radius

    def __repr__(self):
        return f"SphereCollider(center={self.center}, radius={self.radius})"

    def transform_by(self, transform: 'Pose3'):
        """Return a new SphereCollider transformed by the given Pose3."""
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

    def closest_to_union_collider(self, other: "UnionCollider"):
        """Return the closest points and distance between this capsule and a union collider."""
        a,b,c = other.closest_to_collider(self)
        return b,a,c
        
    def closest_to_collider(self, other: "Collider"):
        from .capsule import CapsuleCollider
        from .union_collider import UnionCollider

        """Return the closest points and distance between this collider and another collider."""
        if isinstance(other, SphereCollider):
            return self.closest_to_sphere(other)
        elif isinstance(other, CapsuleCollider):
            return self.closest_to_capsule(other)
        elif isinstance(other, UnionCollider):
            return self.closest_to_union_collider(other)
        else:
            raise NotImplementedError(f"closest_to_collider not implemented for {type(other)}")
