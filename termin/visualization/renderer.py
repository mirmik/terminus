"""Renderer configures OpenGL state and draws entities."""

from __future__ import annotations

from OpenGL import GL as gl

from .camera import CameraComponent, PerspectiveCameraComponent
from .scene import Scene
from .entity import RenderContext

from .opengl_helpers import init_opengl, opengl_is_inited


class Renderer:
    """Responsible for viewport setup, uniforms and draw traversal."""

    def __init__(self):
        pass

    def ensure_ready(self):
        if not opengl_is_inited():
            init_opengl()

    def _ensure_gl_state(self):
        if not opengl_is_inited():
            init_opengl()

    def render_viewport(self, scene: Scene, camera: CameraComponent, viewport_rect: tuple[int, int, int, int], context_key: int):
        self.ensure_ready()
        self._ensure_gl_state()
        x, y, w, h = viewport_rect
        gl.glViewport(x, y, w, h)
        view = camera.get_view_matrix()
        projection = camera.get_projection_matrix()
        context = RenderContext(view=view, projection=projection, camera=camera, scene=scene, renderer=self, context_key=context_key)

        for entity in scene.entities:
            entity.draw(context)
