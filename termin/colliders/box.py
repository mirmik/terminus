from termin.aabb import AABB 
from termin.pose3 import Pose3
import numpy
from termin.colliders.collider import Collider
from termin.geomalgo.project import closest_of_aabb_and_capsule, closest_of_aabb_and_sphere



class BoxCollider(Collider):
    def __init__(self, center : numpy.ndarray, size: numpy.ndarray, pose: Pose3 = Pose3.identity()):
        self.center = center
        self.size = size
        self.pose = pose

    def local_aabb(self) -> AABB:
        half_size = self.size / 2.0
        min_point = self.center - half_size
        max_point = self.center + half_size
        return AABB(min_point, max_point)

    def __repr__(self):
        return f"BoxCollider(center={self.center}, size={self.size}, pose={self.pose})"

    def transform_by(self, tpose: 'Pose3'):
        new_pose = tpose.compose(self.pose)
        return BoxCollider(self.center, self.size, new_pose)

    def point_in_local_frame(self, point: numpy.ndarray) -> numpy.ndarray:
        """Transform point to local frame"""
        return self.pose.inverse_transform_point(point)

    def segment_in_local_frame(self, seg_start: numpy.ndarray, seg_end: numpy.ndarray):
        """Transform segment to local frame"""
        local_start = self.point_in_local_frame(seg_start)
        local_end = self.point_in_local_frame(seg_end)
        return local_start, local_end
    
    def closest_point_to_capsule(self, capsule : "CapsuleCollider"):
        a_local = self.point_in_local_frame(capsule.a)
        b_local = self.point_in_local_frame(capsule.b)
        aabb = self.local_aabb()
        closest_aabb_point, closest_capsule_point, distance = closest_of_aabb_and_capsule(
            aabb.min_point, aabb.max_point,
            a_local, b_local, capsule.radius
        )

        # Transform closest points back to world frame
        closest_aabb_point_world = self.pose.transform_point(closest_aabb_point)
        closest_capsule_point_world = self.pose.transform_point(closest_capsule_point)
        return closest_aabb_point_world, closest_capsule_point_world, distance
    
    def closest_to_sphere(self, sphere : "SphereCollider"):
        c_local = self.point_in_local_frame(sphere.center)
        aabb = self.local_aabb()
        closest_aabb_point, closest_sphere_point, distance = closest_of_aabb_and_sphere(
            aabb.min_point, aabb.max_point,
            c_local, sphere.radius
        )

        # Transform closest points back to world frame
        closest_aabb_point_world = self.pose.transform_point(closest_aabb_point)
        closest_sphere_point_world = self.pose.transform_point(closest_sphere_point)
        return closest_aabb_point_world, closest_sphere_point_world, distance

    def closest_to_collider(self, other: "Collider"):
        from .capsule import CapsuleCollider
        from .sphere import SphereCollider
        if isinstance(other, CapsuleCollider):
            return self.closest_point_to_capsule(other)
        elif isinstance(other, SphereCollider):
            return self.closest_to_sphere(other)
        else:
            raise NotImplementedError(f"closest_to_collider not implemented for {type(other)}")