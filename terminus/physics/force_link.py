#!/usr/bin/env python3

import numpy
from terminus.physics.screw_commutator import VariableValueCommutator
from terminus.ga201.motor import Motor2
from terminus.ga201.screw import Screw2
from terminus.physics.pose_object import ReferencedPoseObject
from terminus.physics.screw_commutator import ScrewCommutator
from terminus.physics.indexed_matrix import IndexedVector


class VariableMultiForceLink:
    def __init__(self, body, coeff, position_in_local_frame):
        self._coeff = coeff
        self._position_in_local_frame = position_in_local_frame
        self._body = body


class VariableMultiForce:
    def __init__(self, position, child, parent, senses=[]):
        self._position_in_child_frame = position * child.position().inverse()
        self._position_in_parent_frame = position * parent.position().inverse()
        self._child = child
        self._parent = parent
        self._senses = senses

        self._pose_object = ReferencedPoseObject(
            parent=child._pose_object, pose=self._position_in_child_frame)
        self._force_screw_variables = ScrewCommutator(
            local_senses=senses, pose_object=self._pose_object)

    def senses(self):
        return self._senses

    def global_position(self):
        return self.global_position_by_parent()

    def global_position_by_parent(self):
        return self._parent.global_position() * self._position_in_parent_frame

    def global_position_by_child(self):
        return self._child.global_position() * self._position_in_child_frame

    def link_body(self, body, coeff, position_in_local_frame):
        self.links[body] = VariableMultiForceLink(
            body, coeff, position_in_local_frame)

    def equation_indexes(self, body):
        return body.equation_indexes()

    def position_in_local_frame(self, body):
        return self._position_in_local_frames[self._bodies.index(body)]

    def B_matrix_list(self):
        dQdl_child = self._force_screw_variables.derivative_matrix_from(
            self._child.acceleration_indexer()).transpose()

        if self._parent is not None:
            dQdl_parent = self._force_screw_variables.derivative_matrix_from(
                self._parent.acceleration_indexer()).transpose()
            return [dQdl_child, dQdl_parent]
        else:
            return [dQdl_child]

    def D_matrix_list(self):
        return IndexedVector(numpy.zeros(len(self._force_screw_variables.indexes())), [self._force_screw_variables.indexes()])


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
