"""Renderer configures graphics state and draws entities."""

from __future__ import annotations

from .camera import CameraComponent, PerspectiveCameraComponent
from .scene import Scene
from .entity import RenderContext
from .backends.base import GraphicsBackend


class Renderer:
    """Responsible for viewport setup, uniforms and draw traversal."""

    def __init__(self, graphics: GraphicsBackend):
        self.graphics = graphics

    def render_viewport(self, scene: Scene, camera: CameraComponent, viewport_rect: tuple[int, int, int, int], context_key: int):
        self.graphics.ensure_ready()
        x, y, w, h = viewport_rect
        self.graphics.set_viewport(x, y, w, h)
        view = camera.get_view_matrix()
        projection = camera.get_projection_matrix()
        context = RenderContext(
            view=view,
            projection=projection,
            camera=camera,
            scene=scene,
            renderer=self,
            context_key=context_key,
            graphics=self.graphics,
        )

        for entity in scene.entities:
            entity.draw(context)
