"""Simple scene graph storing entities and global parameters."""

from __future__ import annotations

from typing import List, Sequence, TYPE_CHECKING

import numpy as np

from .entity import Component, Entity, InputComponent
from .backends.base import GraphicsBackend

if TYPE_CHECKING:  # pragma: no cover
    from .shader import ShaderProgram

def is_overrides_method(obj, method_name, base_class):
    return getattr(obj.__class__, method_name) is not getattr(base_class, method_name)

class Scene:
    """Container for renderable entities and lighting data."""

    def __init__(self, background_color: Sequence[float] = (0.05, 0.05, 0.08, 1.0)):
        self.entities: List[Entity] = []
        self.lights: List[np.ndarray] = []
        self.background_color = np.array(background_color, dtype=np.float32)
        self._shaders_set = set()
        self._inited = False
        self._input_components: List[InputComponent] = []
        self._graphics: GraphicsBackend | None = None

        self.update_list: List[Component] = []

        # Lights
        self.light_direction = np.array([-0.5, -1.0, -0.3], dtype=np.float32)
        self.light_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    def add(self, entity: Entity) -> Entity:
        """Add entity to the scene, keeping the entities list sorted by priority."""
        index = 0
        while index < len(self.entities) and self.entities[index].priority <= entity.priority:
            index += 1
        self.entities.insert(index, entity)
        entity.on_added(self)
        for shader in entity.gather_shaders():
            self._register_shader(shader)
        return entity

    def remove(self, entity: Entity):
        self.entities.remove(entity)
        entity.on_removed()

    def register_component(self, component: Component):
        for shader in component.required_shaders():
            self._register_shader(shader)
        if isinstance(component, InputComponent):
            self._input_components.append(component)
        if is_overrides_method(component, "update", Component):
            self.update_list.append(component)

    def unregister_component(self, component: Component):
        if isinstance(component, InputComponent) and component in self._input_components:
            self._input_components.remove(component)

    def update(self, dt: float):
        for component in self.update_list:
            component.update(dt)

    def ensure_ready(self, graphics: GraphicsBackend):
        if self._inited:
            return
        self._graphics = graphics
        for shader in list(self._shaders_set):
            shader.ensure_ready(graphics)
        self._inited = True

    def _register_shader(self, shader: "ShaderProgram"):
        if shader in self._shaders_set:
            return
        self._shaders_set.add(shader)
        if self._inited and self._graphics is not None:
            shader.ensure_ready(self._graphics)

    def dispatch_input(self, viewport, event: str, **kwargs):
        listeners = list(self._input_components)
        for component in listeners:
            handler = getattr(component, event, None)
            if handler:
                handler(viewport, **kwargs)
