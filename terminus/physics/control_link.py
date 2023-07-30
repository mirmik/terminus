import numpy
from terminus.physics.force_link import VariableMultiForce
from terminus.physics.frame import Frame, ReferencedFrame
from terminus.physics.pose_object import ReferencedPoseObject
from terminus.ga201.screw import Screw2
from terminus.physics.pose_object import ReferencedPoseObject, PoseObject
from terminus.physics.screw_commutator import ScrewCommutator
from terminus.physics.indexed_matrix import IndexedVector

from terminus.physics.extlinalg import outkernel_operator
from terminus.physics.extlinalg import kernel_operator
import math
import time

start_time = 0

class ControlLink(VariableMultiForce):
    def __init__(self, position, child, parent, senses=[], stiffness=[1, 1]):
        super().__init__(position, child, parent, senses, stiffness)
        self._control_vector = [0] * len(self._senses)
        self.curtime = 0

        self.target = numpy.array([0,0])

    def set_control_vector(self, control_vector):
        self._control_vector = control_vector

    def H_matrix_list(self):
        dQdl_child = self.derivative_by_frame(self._child).transpose()
        if self._parent is not None:
            dQdl_parent = -self.derivative_by_frame(self._parent).transpose()
            return [dQdl_child, dQdl_parent]
        else:
            return [dQdl_child]
    
    def Ksi_matrix_list(self, delta, control_tasks):
        lst = []
        for task in control_tasks:
            if self not in task._control_frames:
                continue

            derivative = task.derivative_by_frame(self)
            pinv_derivative = numpy.linalg.pinv(derivative.matrix)

            res = pinv_derivative @ task.control_task(delta)

            if task._filter is not None:
                res = task._filter @ res

            lst.append(IndexedVector(
                res[0],
                idxs=self._screw_commutator.indexes(), 
                comm=self._screw_commutator)
            )
        return lst

class ControlTaskFrame(ReferencedFrame):
    def __init__(self, linked_body, position_in_body):
        senses = [
            #Screw2(m=1),
            Screw2(v=[1,0]),
            Screw2(v=[0,1]),
        ]
        super().__init__(linked_body, position_in_body, senses)
        self.curtime = 0
        self._control_screw = Screw2()
        self._control_frames = []
        self._filter = None

    def set_filter(self, filter):
        self._filter = filter

    def add_control_frame(self, frame):
        self._control_frames.append(frame)

    def control_task(self, delta):
        return self._control_screw.vector()

    def set_control_screw(self, screw):
        rotated_to_local = screw.inverse_rotate_by(self.position())
        self._control_screw = rotated_to_local