
from termin.closest import closest_points_between_segments, closest_points_between_capsules, closest_points_between_capsule_and_sphere
import numpy
from termin.colliders.collider import Collider
from termin.colliders.sphere import SphereCollider

class CapsuleCollider(Collider):
    def closest_to_ray(self, ray: "Ray3"):
        """
        Ближайшие точки между сегментом капсулы и лучом:
        Используем closest_points_between_segments(…)
        Луч считаем как сегмент O + D * t, t >= 0.
        Для упрощения ограничиваем t большим числом.
        """
        from termin.closest import closest_points_between_segments

        O = ray.origin
        D = ray.direction
        FAR = 1e6  # ограничение луча

        ray_a = O
        ray_b = O + D * FAR

        p_seg, p_ray, dist = closest_points_between_segments(
            self.a, self.b,
            ray_a, ray_b
        )

        # Если мы дальше радиуса, то это просто ближайшая точка
        if dist > self.radius:
            p_col = p_seg
            return p_col, p_ray, dist - self.radius

        # Иначе луч пересекает капсулу
        dir_vec = p_ray - p_seg
        n = numpy.linalg.norm(dir_vec)
        if n > 1e-8:
            p_col = p_seg + dir_vec * (self.radius / n)
        else:
            p_col = p_seg
        return p_col, p_ray, max(0.0, numpy.linalg.norm(p_col - p_ray))
    
    def __init__(self, a: numpy.ndarray, b: numpy.ndarray, radius: float):
        self.a = a
        self.b = b
        self.radius = radius

    def transform_by(self, transform: 'Pose3'):
        """Return a new CapsuleCollider transformed by the given Pose3."""
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

        from .sphere import SphereCollider
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
