
from termin.pose3 import Pose3
from termin.colliders.collider import Collider
from termin.transform import Transform3
import numpy

class AttachedCollider:
    """A collider attached to a Transform3 with a local pose."""
    def __init__(self, collider: Collider, transform: 'Transform3', local_pose: Pose3 = Pose3.identity()):
        self._transform = transform
        self._local_pose = local_pose
        self._collider = collider

    def collider(self) -> Collider:
        """Get the collider in world coordinates."""
        world_transform = self._transform.global_pose() * self._local_pose
        return self._collider.transform_by(world_transform)

    def local_pose(self) -> Pose3:
        """Get the local pose of the collider."""
        return self._local_pose

    def transform(self) -> 'Transform3':
        """Get the Transform3 to which this collider is attached."""
        return self._transform

    def distance(self, other: "AttachedCollider") -> float:
        """Return the distance between this attached collider and another attached collider."""
        return self.collider().distance(other.collider())
 
    def closest_to_collider(self, other: "AttachedCollider"):
        """Return the closest points and distance between this attached collider and another attached collider."""
        return self.collider().closest_to_collider(other.collider())

    def avoidance(self, other: "AttachedCollider") -> numpy.ndarray:
        """Compute an avoidance vector to maintain a minimum distance from another attached collider."""
        p_near, q_near, dist = self.closest_to_collider(other)
        diff = p_near - q_near
        real_dist = numpy.linalg.norm(diff)
        if real_dist == 0.0:
            return numpy.zeros(3), 0.0, p_near
        direction = diff / real_dist
        return direction, real_dist, p_near