"""
Базовые геометрические классы (Geometric Base).

Содержит фундаментальные классы для представления геометрии:
- Pose2 - позы (положение + ориентация) в 2D пространстве
- Pose3 - позы (положение + ориентация) в 3D пространстве
- Screw, Screw2, Screw3 - винтовые преобразования
"""

from .pose2 import Pose2
from .pose3 import Pose3
from .screw import Screw, Screw2, Screw3
from .aabb import AABB, TransformAABB

__all__ = [
    'Pose2',
    'Pose3',
    'Screw',
    'Screw2',
    'Screw3',
    'AABB',
    'TransformAABB'
]
