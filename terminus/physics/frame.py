import numpy as np
from terminus.physics.screw_commutator import ScrewCommutator
from terminus.ga201.screw import Screw2
from terminus.physics.pose_object import ReferencedPoseObject, PoseObject

class Frame:
    def __init__(self, pose_object, screws):
        self._pose_object = pose_object
        self._screw_commutator = ScrewCommutator(
            local_senses=screws, pose_object=self._pose_object)

    def screw_commutator(self):
        return self._screw_commutator

    def derivative_by_frame(self, other):
        return self.screw_commutator().derivative_matrix_from(
            other.screw_commutator())

    def global_derivative_by_frame(self, other):
        return self.screw_commutator().global_derivative_matrix_from(
            other.screw_commutator())

    def position(self):
        return self._pose_object.position()

class ReferencedFrame(Frame):
    def __init__(self, linked_body, position_in_body, senses):
        self._parent = linked_body
        pose_object = ReferencedPoseObject(
            parent=linked_body._pose_object, pose=position_in_body)
        super().__init__(pose_object=pose_object, screws=senses)

        self.control = Screw2(v=[0,1])

    def current_position(self):
        return self.position()

    def right_velocity(self):
        parent_right_velocity = self._parent.right_velocity()
        carried = parent_right_velocity.kinematic_carry(
            self._pose_object.relative_position())
        return carried

    def right_velocity_global(self):
        right_velocity = self.right_velocity()
        rotated = right_velocity.rotate_by(self.position())
        return rotated

    def right_acceleration(self):
        parent_right_acceleration = self._parent.right_acceleration()
        carried = parent_right_acceleration.kinematic_carry(
            self._pose_object.relative_position())
        return carried

    def right_acceleration_global(self):
        right_acceleration = self.right_acceleration()
        rotated = right_acceleration.rotate_by(self.position())
        #return rotated
        return self._parent.right_acceleration_global()