import numpy
from terminus.physics.force_link import VariableMultiForce
from terminus.physics.frame import Frame
from terminus.physics.pose_object import ReferencedPoseObject
from terminus.ga201.screw import Screw2
from terminus.physics.pose_object import ReferencedPoseObject, PoseObject
from terminus.physics.screw_commutator import ScrewCommutator
from terminus.physics.indexed_matrix import IndexedVector

from terminus.physics.extlinalg import outkernel_operator
from terminus.physics.extlinalg import kernel_operator
import math
import time

start_time = time.time()

class ControlLink(VariableMultiForce):
    def __init__(self, position, child, parent, senses=[], stiffness=[1, 1], use_child_frame=False):
        super().__init__(position, child, parent, senses, stiffness, use_child_frame)
        self._control_vector = [0] * len(self._senses)

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


        target_vel = Screw2(v=[-1,0])
        current_vel = task.right_velocity_global()
        curpos = task.position()
        curpos = curpos.factorize_translation_vector()
        #current_vel.set_moment(0)

        curtime = time.time()

        s = (math.sin((curtime - start_time)/2))
        c = (math.cos((curtime - start_time)/2))

        target_pos = (numpy.array([10,0]) 
            + (s) * numpy.array([5,0])
            + (c) * numpy.array([0,5])
        )

        errorpos = Screw2(v=target_pos - curpos) * 60 * delta
        target_spd = errorpos

        #target_spd = Screw2(v=[0,0.1])

        errorspd = (target_spd - current_vel) * 30000000  * delta
        error = errorspd

        task_control = error.as_array()

        res = pinv_derivative @ task_control.reshape(3,1)
        #res = derivative.matrix.T @ task.control.as_array().reshape(3,1)

        #print(task.current_position())
        #print(task.right_acceleration())
        #print(task.right_acceleration_global().vector())
        #print(task.right_velocity_global().vector())

        return [IndexedVector(
                #self._control_vector,
                res[0]*0.01,
                idxs=self._screw_commutator.indexes())]


class ControlTaskFrame(Frame):
    def __init__(self, linked_body, position_in_body):
        self._parent = linked_body
        pose_object = ReferencedPoseObject(
            parent=linked_body._pose_object, pose=position_in_body)
        super().__init__(pose_object=pose_object, screws=[
            Screw2(m=1),
            Screw2(v=[1,0]),
            Screw2(v=[0,1]),
        ])

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