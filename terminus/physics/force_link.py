#!/usr/bin/env python3

import numpy
from terminus.physics.screw_commutator import VariableValueCommutator
from terminus.ga201.motor import Motor2
from terminus.ga201.screw import Screw2
from terminus.physics.pose_object import ReferencedPoseObject, PoseObject
from terminus.physics.screw_commutator import ScrewCommutator
from terminus.physics.indexed_matrix import IndexedVector


class VariableMultiForceLink:
    def __init__(self, body, coeff, position_in_local_frame):
        self._coeff = coeff
        self._position_in_local_frame = position_in_local_frame
        self._body = body


class VariableMultiForce:
    def __init__(self, position, child, parent, senses=[], stiffness=[1, 1], use_child_frame=True):
        self._use_child_frame = use_child_frame
        self._position_in_child_frame = child.position().inverse() * position
        if parent is not None:
            self._position_in_parent_frame = parent.position().inverse() * position
            if self._use_child_frame:
                self._pose_object = ReferencedPoseObject(
                    parent=child._pose_object, pose=self._position_in_child_frame)
            else:
                self._pose_object = ReferencedPoseObject(
                    parent=parent._pose_object, pose=self._position_in_parent_frame)
        else:
            self._position_in_parent_frame = position
            if self._use_child_frame:
                self._pose_object = ReferencedPoseObject(
                    parent=child._pose_object, pose=self._position_in_child_frame)
            else:
                self._pose_object = PoseObject(
                    pose=self._position_in_parent_frame)

        self._child = child
        self._parent = parent
        self._senses = senses
        self._last_error = Screw2()
        self._stiffness = stiffness

        self._force_screw_variables = ScrewCommutator(
            local_senses=senses, pose_object=self._pose_object)

    def senses(self):
        return self._senses

    def diff_position(self):
        return self.position_error_screw()

    def global_position(self):
        return self.global_position_by_parent()

    def position_error_motor(self):
        position_as_child = self.global_position_by_child()
        position_as_parent = self.global_position_by_parent()
        # if self._use_child_frame:
        diff = position_as_parent.inverse() * position_as_child
        # else:
        #    diff = position_as_child.inverse() * position_as_parent
        return diff

    def position_error_screw(self):
        return self.position_error_motor().log()

    def global_position_by_parent(self):
        if self._parent is None:
            return self._position_in_parent_frame
        return self._parent.global_position() * self._position_in_parent_frame

    def global_position_by_child(self):
        return self._child.global_position() * self._position_in_child_frame

    def B_matrix_list(self):
        dQdl_child = self._force_screw_variables.derivative_matrix_from(
            self._child.equation_indexer()).transpose()

        if self._parent is not None:
            # Минус из-за того, что в родительском фрейме чувствительность обратна чувствительности в дочернем фрейме
            dQdl_parent = -self._force_screw_variables.derivative_matrix_from(
                self._parent.equation_indexer()).transpose()
            return [dQdl_child, dQdl_parent]
        else:
            return [dQdl_child]

    def D_matrix_list(self, delta):
        error = self.position_error_screw()
        print("Error:", error)
        diff_error = error - self._last_error
        dots = numpy.array([error.fulldot(s)
                            for s in self._senses]) * self._stiffness[0] * 1
        diff_dots = numpy.array([diff_error.fulldot(s)
                                 for s in self._senses]) * self._stiffness[1] * 0
        correction = - dots - diff_dots
        self._last_error = error

        return [IndexedVector(
                correction,
                idxs=self._force_screw_variables.indexes())]


if __name__ == "__main__":
    from terminus.physics.body import Body2
    b1 = Body2()
    b2 = Body2()

    b1.set_position(Motor2.translation(1, 0))
    b2.set_position(Motor2.translation(2, 0))

    fl = VariableMultiForce(Motor2.translation(
        2, 0), b1, b2, senses=[Screw2(m=1), Screw2(v=[1, 0]), Screw2(v=[0, 1])])

    for s in fl.senses():
        print(s)

    B_list = fl.B_matrix_list()
    for B in B_list:
        print(B)
