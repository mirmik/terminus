"""Simple scene graph storing entities and global parameters."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .entity import Entity


class Scene:
    """Container for renderable entities and lighting data."""

    def __init__(self, background_color: Sequence[float] = (0.05, 0.05, 0.08, 1.0)):
        self.entities: List[Entity] = []
        self.lights: List[np.ndarray] = []
        self.background_color = np.array(background_color, dtype=np.float32)
        self._shaders_set = set()
        self._inited = False

        # Lights
        self.light_direction = np.array([-0.5, -1.0, -0.3], dtype=np.float32)

    def add(self, entity: Entity) -> Entity:
        """Add entity to the scene, keeping the entities list sorted by priority."""
        index = 0
        while index < len(self.entities) and self.entities[index].priority <= entity.priority:
            index += 1
        self.entities.insert(index, entity)
        self._shaders_set.add(entity.material.shader)
        return entity

    def remove(self, entity: Entity):
        self.entities.remove(entity)

    def update(self, dt: float):
        for entity in self.entities:
            entity.update(dt)

    def ensure_ready(self):
        if self._inited:
            return
        for shader in self._shaders_set:
            shader.ensure_ready()
        self._inited = True
