
from termin.closest import closest_points_between_segments, closest_points_between_capsules, closest_points_between_capsule_and_sphere
import numpy
from termin.colliders.collider import Collider
from termin.geombase import Pose3



class SphereCollider(Collider):
    def closest_to_ray(self, ray: "Ray3"):
        """
        Аналитическое пересечение луча со сферой.
        Луч: O + D * t
        Центр: C
        Радиус: r
        """
        O = ray.origin
        D = ray.direction
        C = self.center
        r = self.radius

        OC = O - C
        b = 2 * numpy.dot(D, OC)
        c = numpy.dot(OC, OC) - r * r
        disc = b * b - 4 * c

        # Нет пересечения — вернуть ближайшие точки
        if disc < 0:
            # t = -dot(OC, D)
            #t = -numpy.dot(OC, D)
            t = numpy.dot((C - O), D)
            if t < 0:
                t = 0
            p_ray = ray.point_at(t)

            dir_vec = p_ray - C
            dist = numpy.linalg.norm(dir_vec)
            if dist > 1e-8:
                p_col = C + dir_vec * (r / dist)
            else:
                p_col = C + numpy.array([r, 0, 0], dtype=numpy.float32)  # произвольное направление

            return p_col, p_ray, numpy.linalg.norm(p_col - p_ray)

        # Есть пересечения: берем ближайшее t >= 0
        sqrt_disc = numpy.sqrt(disc)
        t1 = (-b - sqrt_disc) * 0.5
        t2 = (-b + sqrt_disc) * 0.5

        t_hit = None
        if t1 >= 0:
            t_hit = t1
        elif t2 >= 0:
            t_hit = t2

        # Пересечение позади луча — перейти к ближайшей точке
        if t_hit is None:
            t = -numpy.dot(OC, D)
            if t < 0:
                t = 0
            p_ray = ray.point_at(t)
            dir_vec = p_ray - C
            dist = numpy.linalg.norm(dir_vec)
            p_col = C + dir_vec * (r / dist)
            return p_col, p_ray, numpy.linalg.norm(p_col - p_ray)

        # Корректное пересечение
        p_ray = ray.point_at(t_hit)
        dir_vec = p_ray - C
        dist = numpy.linalg.norm(dir_vec)
        p_col = C + dir_vec * (r / dist) if dist > 1e-8 else p_ray
        return p_col, p_ray, 0.0
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
