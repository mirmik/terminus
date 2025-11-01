
import numpy

class Collider:
    def transform_by(self, transform: 'Pose3'):
        """Return a new Collider transformed by the given Pose3."""
        raise NotImplementedError("transform_by must be implemented by subclasses.")

    def closest_to_collider(self, other: "Collider"):
        """Return the closest points and distance between this collider and another collider."""
        raise NotImplementedError("closest_to_collider must be implemented by subclasses.")
    
    def avoidance(self, other: "Collider") -> numpy.ndarray:
        """Compute an avoidance vector to maintain a minimum distance from another collider."""
        p_near, q_near, dist = self.closest_to_collider(other)
        diff = p_near - q_near
        real_dist = numpy.linalg.norm(diff)
        if real_dist == 0.0:
            return numpy.zeros(3), 0.0, p_near
        direction = diff / real_dist
        return direction, real_dist, p_near