from termin.colliders.collider import Collider
import numpy

class UnionCollider(Collider):
    def __init__(self, colliders):
        self.colliders = colliders

    def transform_by(self, transform: 'Transform3'):
        """Return a new UnionCollider transformed by the given Transform3."""
        transformed_colliders = [collider.transform_by(transform) for collider in self.colliders]
        return UnionCollider(transformed_colliders)

    def closest_to_collider(self, other: "Collider"):
        """Return the closest points and distance between this union collider and another collider."""
        min_dist = float('inf')
        closest_p = None
        closest_q = None

        for collider in self.colliders:
            p_near, q_near, dist = collider.closest_to_collider(other)
            if dist < min_dist:
                min_dist = dist
                closest_p = p_near
                closest_q = q_near

        return closest_p, closest_q, min_dist