"""Scene entity storing pose, mesh and material references."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from termin.geombase.pose3 import Pose3

from .material import Material
from .mesh import MeshDrawable

from OpenGL import GL as gl


@dataclass
class Entity:
    """Renderable object with transform, geometry and appearance."""

    mesh: MeshDrawable
    material: Material
    pose: Pose3 = field(default_factory=Pose3.identity)
    visible: bool = True
    name: str = "entity"
    scale: float = 1.0
    priority: int = 0  # rendering priority, lower values drawn first

    def model_matrix(self) -> np.ndarray:
        """Construct homogeneous model matrix ``M = [R|t]`` with optional uniform scale."""
        matrix = self.pose.as_matrix().copy()
        matrix[:3, :3] *= self.scale
        return matrix

    def update(self, dt: float):
        """Override to implement per-frame animation."""
        # Prototype hook, intentionally empty.
        return

    def draw(self):
        """Draw the entity's mesh."""
        gl.glDepthFunc(gl.GL_LESS)
        self.mesh.draw()