"""Scene entity storing components (Unity-like architecture)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Type, TypeVar, TYPE_CHECKING

import numpy as np

from termin.geombase.pose3 import Pose3
from .backends.base import GraphicsBackend

if TYPE_CHECKING:  # pragma: no cover
    from .camera import Camera
    from .renderer import Renderer
    from .scene import Scene
    from .shader import ShaderProgram


@dataclass
class RenderContext:
    """Data bundle passed to components during rendering."""

    view: np.ndarray
    projection: np.ndarray
    camera: "Camera"
    scene: "Scene"
    renderer: "Renderer"
    context_key: int
    graphics: GraphicsBackend


class Component:
    """Base class for all entity components."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.entity: Optional["Entity"] = None
        self._started = False

    def required_shaders(self) -> Iterable["ShaderProgram"]:
        """Return shaders that must be compiled before rendering."""
        return ()

    def start(self, scene: "Scene"):
        """Called once when the component becomes part of an active scene."""
        self._started = True

    def update(self, dt: float):
        """Called every frame."""
        return

    def draw(self, context: RenderContext):
        """Issue draw calls."""
        return

    def on_removed(self):
        """Called when component is removed from its entity."""
        return


class InputComponent(Component):
    """Component capable of handling input events."""

    def on_mouse_button(self, viewport, button: int, action: int, mods: int):
        return

    def on_mouse_move(self, viewport, x: float, y: float, dx: float, dy: float):
        return

    def on_scroll(self, viewport, xoffset: float, yoffset: float):
        return

    def on_key(self, viewport, key: int, scancode: int, action: int, mods: int):
        return


C = TypeVar("C", bound=Component)


@dataclass
class Entity:
    """Container of components with transform data."""

    pose: Pose3 = field(default_factory=Pose3.identity)
    visible: bool = True
    active: bool = True
    name: str = "entity"
    scale: float = 1.0
    priority: int = 0  # rendering priority, lower values drawn first

    def __post_init__(self):
        self.scene: Optional["Scene"] = None
        self._components: List[Component] = []

    def model_matrix(self) -> np.ndarray:
        """Construct homogeneous model matrix ``M = [R|t]`` with optional uniform scale."""
        matrix = self.pose.as_matrix().copy()
        matrix[:3, :3] *= self.scale
        return matrix

    def add_component(self, component: Component) -> Component:
        component.entity = self
        self._components.append(component)
        if self.scene is not None:
            self.scene.register_component(component)
            if not component._started:
                component.start(self.scene)
        return component

    def remove_component(self, component: Component):
        if component not in self._components:
            return
        self._components.remove(component)
        if self.scene is not None:
            self.scene.unregister_component(component)
        component.on_removed()
        component.entity = None

    def get_component(self, component_type: Type[C]) -> Optional[C]:
        for comp in self._components:
            if isinstance(comp, component_type):
                return comp
        return None

    @property
    def components(self) -> List[Component]:
        return list(self._components)

    def update(self, dt: float):
        if not self.active:
            return
        for component in self._components:
            if component.enabled:
                component.update(dt)

    def draw(self, context: RenderContext):
        if not (self.active and self.visible):
            return
        for component in self._components:
            if component.enabled:
                component.draw(context)

    def gather_shaders(self) -> Iterable["ShaderProgram"]:
        for component in self._components:
            yield from component.required_shaders()

    def on_added(self, scene: "Scene"):
        self.scene = scene
        for component in self._components:
            scene.register_component(component)
            if not component._started:
                component.start(scene)

    def on_removed(self):
        for component in self._components:
            if self.scene is not None:
                self.scene.unregister_component(component)
            component.on_removed()
            component.entity = None
        self.scene = None
