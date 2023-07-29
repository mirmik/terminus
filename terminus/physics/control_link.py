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
    def __init__(self, position, child, parent, senses=[], stiffness=[1, 1], use_child_frame=False):
        super().__init__(position, child, parent, senses, stiffness, use_child_frame)
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
        task = control_tasks[0]
        derivative = task.global_derivative_by_frame(self)
        pinv_derivative = numpy.linalg.pinv(derivative.matrix)

        res = pinv_derivative @ task.control_task(delta).reshape(3,1)

        return [IndexedVector(
                res[0],
                idxs=self._screw_commutator.indexes())]


class ControlTaskFrame(ReferencedFrame):
    def __init__(self, linked_body, position_in_body):
        senses = [
            Screw2(m=1),
            Screw2(v=[1,0]),
            Screw2(v=[0,1]),
        ]
        super().__init__(linked_body, position_in_body, senses)
        self.curtime = 0

    def control_task(self, delta):
        current_vel = self.right_velocity_global()
        curpos = self.position()
        curpos = curpos.factorize_translation_vector()

        curtime = self.curtime
        self.curtime += delta

        D = 1

        s = (math.sin((curtime - start_time)/D))
        c = (math.cos((curtime - start_time)/D))

        ds = (math.cos((curtime - start_time)/D))/D
        dc = -(math.sin((curtime - start_time)/D))/D

        d2s = -(math.sin((curtime - start_time)/D))/D/D
        d2c = -(math.cos((curtime - start_time)/D))/D/D
        
        A = 5
        B = 9

        target_pos = (numpy.array([10,0]) 
            + (s) * numpy.array([A,0])
            + (c) * numpy.array([0,B])
        )
        target_vel =( (ds) * numpy.array([A,0])
            + (dc) * numpy.array([0,B])
        )
        target_acc =( (d2s) * numpy.array([A,0])
            + (d2c) * numpy.array([0,B]))

        k = curtime / 10
        self.target = target_pos

        errorpos = Screw2(v=target_pos - curpos)
        control_spd = errorpos * 5 + Screw2(v=target_vel)
        errorspd = (control_spd - current_vel)
        erroracc = errorspd * 80 +  Screw2(v=target_acc)

        task_control = erroracc.as_array()
        return task_control