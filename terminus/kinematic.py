from terminus.transform import Transform3
from terminus.pose3 import Pose3
from terminus.screw import Screw3
import numpy

class KinematicTransform3(Transform3):
    """A Transform3 specialized for kinematic chains."""
    def __init__(self, parent: Transform3 = None):
        super().__init__(parent=parent)
        self.output = Transform3(parent=self)
    
    def senses() -> [Screw3]:
        """Return the list of screws representing the sensitivities of this kinematic transform."""
        raise NotImplementedError("senses method must be implemented by subclasses.")

    def link(self, child: 'Transform3'):
        """Link a child Transform3 to this KinematicTransform3 output."""
        self.output.add_child(child)

class KinematicTransform3OneScrew(KinematicTransform3):
    """A Transform3 specialized for 1-DOF kinematic chains."""
    def __init__(self, parent: Transform3 = None):
        super().__init__(parent=parent)
        self._sens = None  # To be defined in subclasses

    def sensitivity_for_basis(self, basis: numpy.ndarray) -> Screw3:
        """Описывает, как влияет изменение координаты влияет на тело связанное с системой basis в системе отсчета самого basis."""
        my_pose = self.global_pose()
        my_pose_in_basis = basis.inverse() * my_pose
        return self._sens.transform_as_twist_by(my_pose_in_basis)

    def senses(self) -> [Screw3]:
        return [self._sens]

    def senses_for_basis(self, basis: numpy.ndarray) -> [Screw3]:
        return [self.sensitivity_for_basis(basis)]

    def sensivity(self) -> Screw3:
        """Return the screw representing the sensitivity of this kinematic transform."""
        return self._sens

    def set_coord(self, coord: float):
        """Set the coordinate of this kinematic transform."""
        self.output.relocate((self._sens * coord).as_pose3())
    

class Rotator3(KinematicTransform3OneScrew):
    def __init__(self, axis: numpy.ndarray, parent: Transform3 = None):
        """Initialize a Rotator that rotates around a given axis by angle_rad."""
        super().__init__(parent=parent)
        self._sens = Screw3(ang=axis, lin=numpy.array([0.0, 0.0, 0.0]))

class Actuator3(KinematicTransform3OneScrew):
    def __init__(self, axis: numpy.ndarray, parent: Transform3 = None):
        """Initialize an Actuator that moves along a given screw."""
        super().__init__(parent=parent)
        self._sens = Screw3(ang=axis, lin=numpy.array([0.0, 0.0, 0.0]))

