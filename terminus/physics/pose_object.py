import numpy
from terminus.ga201.motor import Motor2


class PoseObject:
    def __init__(self, pose=Motor2()):
        self._position = pose

    def position(self):
        return self._position

    def update_position(self, pose):
        self._position = pose

    def self_normalize(self):
        self._position.self_normalize()


class ReferencedPoseObject:
    def __init__(self, pose=Motor2(), parent=None):
        self._pose_in_frame = pose
        self._parent = parent

    def position(self):
        return self._parent.position() * self._pose_in_frame
