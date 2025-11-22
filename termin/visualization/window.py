"""Window abstraction delegating platform details to a backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .camera import CameraComponent
from .renderer import Renderer
from .scene import Scene
from .backends.base import (
    Action,
    GraphicsBackend,
    Key,
    MouseButton,
    WindowBackend,
    BackendWindow,
)


@dataclass
class Viewport:
    scene: Scene
    camera: CameraComponent
    rect: Tuple[float, float, float, float]
    canvas: Optional["Canvas"] = None


class Window:
    """Manages a platform window and a set of viewports."""

    def __init__(self, width: int, height: int, title: str, renderer: Renderer, graphics: GraphicsBackend, window_backend: WindowBackend, share=None, **backend_kwargs):
        self.renderer = renderer
        self.graphics = graphics
        share_handle = None
        if isinstance(share, Window):
            share_handle = share.handle
        elif isinstance(share, BackendWindow):
            share_handle = share

        self.window_backend = window_backend
        self.handle: BackendWindow = self.window_backend.create_window(width, height, title, share=share_handle, **backend_kwargs)

        self.viewports: List[Viewport] = []
        self._active_viewport: Optional[Viewport] = None
        self._last_cursor: Optional[Tuple[float, float]] = None

        self.handle.set_user_pointer(self)
        self.handle.set_framebuffer_size_callback(self._handle_framebuffer_resize)
        self.handle.set_cursor_pos_callback(self._handle_cursor_pos)
        self.handle.set_scroll_callback(self._handle_scroll)
        self.handle.set_mouse_button_callback(self._handle_mouse_button)
        self.handle.set_key_callback(self._handle_key)

    def close(self):
        if self.handle:
            self.handle.close()
            self.handle = None

    @property
    def should_close(self) -> bool:
        return self.handle is None or self.handle.should_close()

    def make_current(self):
        if self.handle is not None:
            self.handle.make_current()

    def add_viewport(self, scene: Scene, camera: CameraComponent, rect: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0), canvas: Optional[Canvas] = None) -> Viewport:
        if not self.handle.drives_render():
            self.make_current()
        scene.ensure_ready(self.graphics)
        viewport = Viewport(scene=scene, camera=camera, rect=rect, canvas=canvas)
        camera.viewport = viewport
        self.viewports.append(viewport)
        return viewport

    def update(self, dt: float):
        # Reserved for future per-window updates.
        return

    def render(self):
        self._render_core(from_backend=False)


    # Event handlers -----------------------------------------------------

    def _handle_framebuffer_resize(self, window, width, height):
        return

    def _handle_mouse_button(self, window, button: MouseButton, action: Action, mods):
        if self.handle is None:
            return
        x, y = self.handle.get_cursor_pos()
        viewport = self._viewport_under_cursor(x, y)
        if action == Action.PRESS:
            self._active_viewport = viewport
        if action == Action.RELEASE:
            self._last_cursor = None
            if viewport is None:
                viewport = self._active_viewport
            self._active_viewport = None
        if viewport is not None:
            viewport.scene.dispatch_input(viewport, "on_mouse_button", button=button, action=action, mods=mods)

    def _handle_cursor_pos(self, window, x, y):
        if self.handle is None:
            return
        if self._last_cursor is None:
            dx = dy = 0.0
        else:
            dx = x - self._last_cursor[0]
            dy = y - self._last_cursor[1]
        self._last_cursor = (x, y)
        viewport = self._active_viewport or self._viewport_under_cursor(x, y)
        if viewport is not None:
            viewport.scene.dispatch_input(viewport, "on_mouse_move", x=x, y=y, dx=dx, dy=dy)

    def _handle_scroll(self, window, xoffset, yoffset):
        if self.handle is None:
            return
        x, y = self.handle.get_cursor_pos()
        viewport = self._viewport_under_cursor(x, y) or self._active_viewport
        if viewport is not None:
            viewport.scene.dispatch_input(viewport, "on_scroll", xoffset=xoffset, yoffset=yoffset)

    def _handle_key(self, window, key: Key, scancode: int, action: Action, mods):
        if key == Key.ESCAPE and action == Action.PRESS and self.handle is not None:
            self.handle.set_should_close(True)
        viewport = self._active_viewport or (self.viewports[0] if self.viewports else None)
        if viewport is not None:
            viewport.scene.dispatch_input(viewport, "on_key", key=key, scancode=scancode, action=action, mods=mods)

    def _viewport_under_cursor(self, x: float, y: float) -> Optional[Viewport]:
        if self.handle is None or not self.viewports:
            return None
        win_w, win_h = self.handle.window_size()
        if win_w == 0 or win_h == 0:
            return None
        nx = x / win_w
        ny = 1.0 - (y / win_h)
        for viewport in self.viewports:
            vx, vy, vw, vh = viewport.rect
            if vx <= nx <= vx + vw and vy <= ny <= vy + vh:
                return viewport
        return None

    def _render_core(self, from_backend: bool):
        if self.handle is None:
            return

        self.graphics.ensure_ready()
        
        if not from_backend:
            self.make_current()

        context_key = id(self)
        width, height = self.handle.framebuffer_size()

        for viewport in self.viewports:
            vx, vy, vw, vh = viewport.rect
            px = int(vx * width)
            py = int(vy * height)
            pw = max(1, int(vw * width))
            ph = max(1, int(vh * height))

            viewport.camera.set_aspect(pw / max(1.0, float(ph)))

            self.graphics.enable_scissor(px, py, pw, ph)
            bg = viewport.scene.background_color
            self.graphics.clear_color_depth(bg)
            self.graphics.disable_scissor()

            self.renderer.render_viewport(
                viewport.scene, viewport.camera,
                (px, py, pw, ph),
                context_key
            )

            if viewport.canvas:
                viewport.canvas.render(self.graphics, context_key, (px, py, pw, ph))

        # GLFW — делает swap
        if not from_backend:
            self.handle.swap_buffers()



# Backwards compatibility
GLWindow = Window
