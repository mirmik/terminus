"""GLFW-based window wrapper orchestrating update and render loops."""

from __future__ import annotations

import time

import glfw

from .camera import OrbitCamera
from .renderer import Renderer
from .scene import Scene


class GLWindow:
    """High-level window abstraction that glues together scene, camera and renderer."""

    def __init__(self, width: int = 1280, height: int = 720, title: str = "termin viewer", scene: Scene | None = None, renderer: Renderer | None = None, camera: OrbitCamera | None = None):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

        self.scene = scene or Scene()
        aspect = width / float(height)
        self.camera = camera or OrbitCamera(radius=6.0, azimuth=45.0, elevation=30.0, aspect=aspect)
        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(self.window)

        if renderer is None:
            raise RuntimeError("Renderer must be provided to GLWindow")

        self.renderer = renderer
        self.renderer.ensure_ready()
        self.scene.ensure_ready()
        glfw.set_window_user_pointer(self.window, self)
        glfw.set_framebuffer_size_callback(self.window, self._handle_framebuffer_resize)
        glfw.set_cursor_pos_callback(self.window, self._handle_cursor_pos)
        glfw.set_scroll_callback(self.window, self._handle_scroll)
        glfw.set_mouse_button_callback(self.window, self._handle_mouse_button)

        self._orbit_speed = 0.2
        self._pan_speed = 0.005
        self._zoom_speed = 0.5
        self._mouse_left = False
        self._mouse_right = False
        self._last_cursor = None
        width, height = glfw.get_framebuffer_size(self.window)
        self.renderer.resize(width, height, self.camera)

    def _handle_framebuffer_resize(self, window, width, height):
        if height == 0:
            height = 1
        self.renderer.resize(width, height, self.camera)

    def _handle_mouse_button(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._mouse_left = action == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._mouse_right = action == glfw.PRESS
        if action == glfw.RELEASE:
            self._last_cursor = None

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
        if self._mouse_left:
            self.camera.orbit(- dx * self._orbit_speed, dy * self._orbit_speed)
        elif self._mouse_right:
            self.camera.pan(-dx * self._pan_speed, dy * self._pan_speed)

    def _handle_scroll(self, window, xoffset, yoffset):
        self.camera.zoom(-yoffset * self._zoom_speed)

    def _handle_keyboard(self, dt: float):
        move = dt * 1.0
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            self.camera.zoom(-move)
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
            self.camera.zoom(move)
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            self.camera.orbit(-60.0 * dt, 0.0)
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            self.camera.orbit(60.0 * dt, 0.0)
        if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)

    def run(self):
        last = time.perf_counter()
        while not glfw.window_should_close(self.window):
            now = time.perf_counter()
            dt = now - last
            last = now
            self._handle_keyboard(dt)
            self.scene.update(dt)
            self.renderer.render(self.scene, self.camera)
            glfw.swap_buffers(self.window)
            glfw.poll_events()
        glfw.terminate()
