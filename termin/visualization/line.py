# skybox.py

from __future__ import annotations
import numpy as np

from termin.geombase.pose3 import Pose3

from .entity import Entity
from termin.mesh.mesh import Mesh
from .mesh import MeshDrawable
from .material import Material
from .shader import ShaderProgram

#gl
from OpenGL import GL as gl

from termin.visualization.polyline import Polyline, PolylineDrawable


class LineEntity(Entity):
    def __init__(self, points: list[np.ndarray], color: np.ndarray = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32), width: float = 1.0, size: float = 1.0, material: Optional[Material] = None):
        mesh = PolylineDrawable(Polyline(vertices=np.array(points, dtype=np.float32),
                        indices=np.array(range(len(points)), dtype=np.uint32)))

        if material is None:
            raise ValueError("Material must be provided for LineEntity.")

        super().__init__(
            mesh=mesh,
            material=material,
            pose=Pose3.identity(),
            scale=size,
            name="line",
            priority = -100,  # рисуем в самом начале
        )

    def draw(self):
        self.mesh.draw()