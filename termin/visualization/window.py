"""GLFW window wrapper supporting multiple viewports and cameras."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import glfw
from OpenGL import GL as gl

from .camera import CameraComponent, CameraController, OrbitCameraController
from .entity import Component
from .renderer import Renderer
from .scene import Scene


def _ensure_glfw():
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")


@dataclass
class Viewport:
    scene: Scene
    camera: CameraComponent
    rect: Tuple[float, float, float, float]
    controller: Optional[CameraController] = None


class GLWindow:
    """Manages a GLFW window and a set of viewports."""

    def __init__(self, width: int, height: int, title: str, renderer: Renderer, share=None):
        _ensure_glfw()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

        self.renderer = renderer
        share_handle = None
        if isinstance(share, GLWindow):
            share_handle = share.window
        elif share is not None:
            share_handle = share

        self.window = glfw.create_window(width, height, title, None, share_handle)
        if not self.window:
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(self.window)
        self.viewports: List[Viewport] = []
        self._active_viewport: Optional[Viewport] = None
        self._mouse_left = False
        self._mouse_right = False
        self._last_cursor: Optional[Tuple[float, float]] = None

        glfw.set_window_user_pointer(self.window, self)
        glfw.set_framebuffer_size_callback(self.window, self._handle_framebuffer_resize)
        glfw.set_cursor_pos_callback(self.window, self._handle_cursor_pos)
        glfw.set_scroll_callback(self.window, self._handle_scroll)
        glfw.set_mouse_button_callback(self.window, self._handle_mouse_button)
        glfw.set_key_callback(self.window, self._handle_key)

    def close(self):
        if self.window:
            glfw.destroy_window(self.window)
            self.window = None

    @property
    def should_close(self) -> bool:
        return self.window is None or glfw.window_should_close(self.window)

    def make_current(self):
        if self.window is not None:
            glfw.make_context_current(self.window)

    def add_viewport(self, scene: Scene, camera: CameraComponent, rect: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)) -> Viewport:
        self.make_current()
        scene.ensure_ready()
        controller = None
        if camera.entity is not None:
            controller = camera.entity.get_component(CameraController)
            if controller is None:
                controller = camera.entity.get_component(OrbitCameraController)
        viewport = Viewport(scene=scene, camera=camera, rect=rect, controller=controller)
        self.viewports.append(viewport)
        return viewport

    def update(self, dt: float):
        # Reserved for future per-window updates.
        return

    def render(self):
        if self.window is None:
            return
        self.make_current()
        width, height = glfw.get_framebuffer_size(self.window)
        for viewport in self.viewports:
            vx, vy, vw, vh = viewport.rect
            px = int(vx * width)
            py = int(vy * height)
            pw = max(1, int(vw * width))
            ph = max(1, int(vh * height))
            gl_viewport_y = py
            viewport.camera.set_aspect(pw / max(1.0, float(ph)))
            gl.glEnable(gl.GL_SCISSOR_TEST)
            gl.glScissor(px, gl_viewport_y, pw, ph)
            bg = viewport.scene.background_color
            gl.glClearColor(float(bg[0]), float(bg[1]), float(bg[2]), float(bg[3]))
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glDisable(gl.GL_SCISSOR_TEST)
            self.renderer.render_viewport(viewport.scene, viewport.camera, (px, gl_viewport_y, pw, ph), self)
        glfw.swap_buffers(self.window)

    # Event handlers -----------------------------------------------------

    def _handle_framebuffer_resize(self, window, width, height):
        return

    def _handle_mouse_button(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._mouse_left = action == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._mouse_right = action == glfw.PRESS
        if action == glfw.PRESS:
            x, y = glfw.get_cursor_pos(self.window)
            self._active_viewport = self._viewport_under_cursor(x, y)
        if action == glfw.RELEASE:
            self._last_cursor = None
            if not (self._mouse_left or self._mouse_right):
                self._active_viewport = None

    def _handle_cursor_pos(self, window, x, y):
        if not (self._mouse_left or self._mouse_right):
            self._last_cursor = (x, y)
            return
        if self._last_cursor is None:
            self._last_cursor = (x, y)
            return
        dx = x - self._last_cursor[0]
        dy = y - self._last_cursor[1]
        self._last_cursor = (x, y)
        viewport = self._active_viewport or self._viewport_under_cursor(x, y)
        if viewport is None or viewport.controller is None:
            return
        if self._mouse_left:
            viewport.controller.orbit(-dx * 0.2, dy * 0.2)
        elif self._mouse_right:
            viewport.controller.pan(-dx * 0.005, dy * 0.005)

    def _handle_scroll(self, window, xoffset, yoffset):
        x, y = glfw.get_cursor_pos(self.window)
        viewport = self._viewport_under_cursor(x, y)
        if viewport and viewport.controller:
            viewport.controller.zoom(-yoffset * 0.5)

    def _handle_key(self, window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)

    def _viewport_under_cursor(self, x: float, y: float) -> Optional[Viewport]:
        if self.window is None or not self.viewports:
            return None
        win_w, win_h = glfw.get_window_size(self.window)
        if win_w == 0 or win_h == 0:
            return None
        nx = x / win_w
        ny = 1.0 - (y / win_h)
        for viewport in self.viewports:
            vx, vy, vw, vh = viewport.rect
            if vx <= nx <= vx + vw and vy <= ny <= vy + vh:
                return viewport
        return None
