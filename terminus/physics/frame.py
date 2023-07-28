import numpy as np
from terminus.physics.pose_object import PoseObject
from terminus.physics.screw_commutator import ScrewCommutator
from terminus.ga201.screw import Screw2

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