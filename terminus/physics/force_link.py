#!/usr/bin/env python3

import numpy
from terminus.physics.screw_commutator import VariableValueCommutator
from terminus.ga201.motor import Motor2
from terminus.ga201.screw import Screw2
from terminus.physics.pose_object import ReferencedPoseObject, PoseObject
from terminus.physics.screw_commutator import ScrewCommutator
from terminus.physics.indexed_matrix import IndexedVector
from terminus.physics.frame import Frame


#class VariableMultiForceLink:
#    def __init__(self, body, coeff, position_in_local_frame):
#        self._coeff = coeff
#        self._position_in_local_frame = position_in_local_frame
#        self._body = body


class VariableMultiForce(Frame):
    def __init__(self, position, child, parent, senses=[], stiffness=[1, 1], use_child_frame=False):
        self._use_child_frame = use_child_frame
        if self._use_child_frame is not False:
            raise Exception("Not implemented")
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

        super().__init__(pose_object=self._pose_object, screws=senses)
        
        self._child = child
        self._parent = parent
        self._senses = senses
        self._stiffness = stiffness

    def senses(self):
        return self._senses

    def diff_position(self):
        return self.position_error_screw()

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

    def velocity_error_screw(self):
        parent_velocity = self.frame_velocity_by_parent()
        #print("parent_velocity:", parent_velocity)
        child_velocity = self.frame_velocity_by_child()
        #print("child_velocity:", child_velocity)
        return child_velocity - parent_velocity
        #child_velocity - parent_velocity

    def parrent_velocity_correction(self, poserror, velerror):
        if self._parent is None:
            return
        parent_velocity = (self._parent.right_velocity()
                .inverted_kinematic_carry(self._position_in_child_frame))

        parent_velocity_correction = parent_velocity + velerror / 2
        self._parent.set_right_velocity(
            parent_velocity_correction.kinematic_carry(self._position_in_parent_frame))

    def child_velocity_correction(self, poserror, velerror):
        child_velocity = (self._child.right_velocity()
                .inverted_kinematic_carry(self._position_in_child_frame)
                .kinematic_carry(poserror))
        
        mul = 1 if self._parent is None else 0.5
        child_velocity_corrected = child_velocity - velerror * mul

        returned_velocity = (
                child_velocity_corrected
                .inverted_kinematic_carry(poserror)
                .kinematic_carry(self._position_in_child_frame)
                )

        #return
        self._child.set_right_velocity(returned_velocity)

    def child_position_correction(self, poserrorm):
        poserror = -self.position_error_screw()
        poserror.set_moment(0)
        #translation = poserror.factorize_translation_vector()
        #rotation = poserror.factorize_rotation_angle()

        # motor from translation
        #err = Motor2.translation(translation[0], translation[1])
        #child_motor = self._child.global_position() * self._position_in_child_frame
        #child_motor = child_motor * err.inverse()
        #poserror = poserror.kinematic_carry(poserrorm)
        poserror_in_child_frame = poserror.inverted_kinematic_carry(
            self._position_in_child_frame)
        print("poserror:", poserror)
        print("poserror_in_child_frame:", poserror_in_child_frame)
        self._child.set_position(self._child.position() * Motor2.from_screw(poserror_in_child_frame)) 
        
        print("poserror after:", -self.position_error_screw())
        
    def velocity_correction(self):
        poserror = self.position_error_motor()
        velerror = self.velocity_error_screw()
        velerror.set_moment(0)

        #raise Exception("Not implemented")
        #self.parrent_velocity_correction(poserror,velerror)
        #self.child_velocity_correction(poserror,velerror)
        #self.child_position_correction(poserror)
    

    def frame_velocity_by_parent(self):
        if self._parent is None:
            return Screw2()
        
        vel = self._parent.right_velocity()
        res = (vel
            .inverted_kinematic_carry(self._position_in_child_frame)
        )
        return vel

    def frame_velocity_by_child(self):
        diff = self.position_error_motor()
        vel = self._child.right_velocity()
        res = (vel
            .inverted_kinematic_carry(self._position_in_child_frame)
            .kinematic_carry(diff)
        )
        return res

    def global_position_by_parent(self):
        if self._parent is None:
            return self._position_in_parent_frame
        return self._parent.global_position() * self._position_in_parent_frame

    def global_position_by_child(self):
        return self._child.global_position() * self._position_in_child_frame

    def B_matrix_list(self):
        dQdl_child = self.derivative_by_frame(self._child).transpose()

        if self._parent is not None:
            # Минус из-за того, что в родительском фрейме чувствительность обратна чувствительности в дочернем фрейме
            dQdl_parent = -self.derivative_by_frame(self._parent).transpose()
            return [dQdl_child, dQdl_parent]
        else:
            return [dQdl_child]

    def D_matrix_list(self, delta):
        poserror = self.position_error_screw()
        velerror = self.velocity_error_screw()
        #print("poserror:", poserror)
        #print("velerror:", velerror)
        posdots = numpy.array([poserror.fulldot(s)
                            for s in self._senses]) * self._stiffness[0] * 200 
        veldots = numpy.array([velerror.fulldot(s)
                            for s in self._senses]) * self._stiffness[1] * 150
        correction = - posdots - veldots
        return [IndexedVector(
                correction,
                idxs=self._screw_commutator.indexes())]


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
