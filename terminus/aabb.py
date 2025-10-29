
import numpy 


class AABB:
    """Axis-Aligned Bounding Box in 3D space."""

    def __init__(self, min_point: numpy.ndarray, max_point: numpy.ndarray):
        self.min_point = min_point
        self.max_point = max_point

    def extend(self, point: numpy.ndarray):
        """Extend the AABB to include the given point."""
        self.min_point = numpy.minimum(self.min_point, point)
        self.max_point = numpy.maximum(self.max_point, point)

    def intersects(self, other: "AABB") -> bool:
        """Check if this AABB intersects with another AABB."""
        return numpy.all(self.max_point >= other.min_point) and numpy.all(other.max_point >= self.min_point)

    def __repr__(self):
        return f"AABB(min_point={self.min_point}, max_point={self.max_point})"

    def from_points(points: numpy.ndarray) -> "AABB":
        """Create an AABB that encompasses a set of points."""
        min_point = numpy.min(points, axis=0)
        max_point = numpy.max(points, axis=0)
        return AABB(min_point, max_point)

    def merge(self, other: "AABB") -> "AABB":
        """Merge this AABB with another AABB and return the resulting AABB."""
        new_min = numpy.minimum(self.min_point, other.min_point)
        new_max = numpy.maximum(self.max_point, other.max_point)
        return AABB(new_min, new_max)