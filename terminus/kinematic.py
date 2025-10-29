from terminus.transform import Transform3
from terminus.pose3 import Pose3
from terminus.screw import Screw3
import numpy

class KinematicTransform3(Transform3):
    """A Transform3 specialized for kinematic chains."""
    def __init__(self, parent: Transform3 = None):
        super().__init__(parent=None)
        self.output = Transform3(parent=self)
        self.kinematic_parent = None
        if parent:
            parent.link(self)
    
    def senses() -> [Screw3]:
        """Return the list of screws representing the sensitivities of this kinematic transform.
        Для совместимости с KinematicChain3 чувствительности возвращаются в порядке от дистального к проксимальному."""
        raise NotImplementedError("senses method must be implemented by subclasses.")

    def link(self, child: 'Transform3'):
        """Link a child Transform3 to this KinematicTransform3 output."""
        self.output.add_child(child)

    @staticmethod
    def found_first_kinematic_unit_in_parent_tree(body, ignore_self: bool = True) -> 'KinematicTransform3':
        if ignore_self:
            body = body.parent

        link = body
        while link is not None:
            if isinstance(link, KinematicTransform3):
                return link
            link = link.parent
        return None

    def update_kinematic_parent(self):
        """Update the kinematic parent of this transform."""
        self.kinematic_parent = KinematicTransform3.found_first_kinematic_unit_in_parent_tree(self, ignore_self=True)

    def update_kinematic_parent_recursively(self):
        """Recursively update the kinematic parent for this transform and its children."""
        self.update_kinematic_parent()
        if self.kinematic_parent is not None:
            self.kinematic_parent.update_kinematic_parent_recursively()

class KinematicTransform3OneScrew(KinematicTransform3):
    """A Transform3 specialized for 1-DOF kinematic chains."""
    def __init__(self, parent: Transform3 = None):
        super().__init__(parent=parent)
        self._sens = None  # To be defined in subclasses
        self._coord = 0.0  # Current coordinate value

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
        self._coord = coord

    def get_coord(self) -> float:
        """Get the current coordinate of this kinematic transform."""
        return self._coord
    

class Rotator3(KinematicTransform3OneScrew):
    def __init__(self, axis: numpy.ndarray, parent: Transform3 = None):
        """Initialize a Rotator that rotates around a given axis by angle_rad."""
        super().__init__(parent=parent)
        self._sens = Screw3(ang=axis, lin=numpy.array([0.0, 0.0, 0.0]))

class Actuator3(KinematicTransform3OneScrew):
    def __init__(self, axis: numpy.ndarray, parent: Transform3 = None):
        """Initialize an Actuator that moves along a given screw."""
        super().__init__(parent=parent)
        self._sens = Screw3(lin=axis, ang=numpy.array([0.0, 0.0, 0.0]))

