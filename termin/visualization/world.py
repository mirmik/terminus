"""Visualization world orchestrating scenes, windows and main loop."""

from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

import glfw

from .renderer import Renderer
from .scene import Scene
from .window import GLWindow


class VisualizationWorld:
    """High-level application controller."""

    def __init__(self):
        self.renderer = Renderer()
        self.scenes: List[Scene] = []
        self.windows: List[GLWindow] = []
        self._running = False
        
        self.fps = 0

    def add_scene(self, scene: Scene) -> Scene:
        self.scenes.append(scene)
        return scene

    def remove_scene(self, scene: Scene):
        if scene in self.scenes:
            self.scenes.remove(scene)

    def create_window(self, width: int = 1280, height: int = 720, title: str = "termin viewer") -> GLWindow:
        share = self.windows[0] if self.windows else None
        window = GLWindow(width=width, height=height, title=title, renderer=self.renderer, share=share)
        self.windows.append(window)
        return window

    def add_window(self, window: GLWindow):
        self.windows.append(window)

    def update_fps(self, dt):
        if dt > 0:
            self.fps = int(1.0 / dt)
        else:
            self.fps = 0

    def run(self):
        if self._running:
            return
        self._running = True
        last = time.perf_counter()

        while self.windows:
            now = time.perf_counter()
            dt = now - last
            last = now

            for scene in list(self.scenes):
                scene.update(dt)

            alive = []
            for window in list(self.windows):
                if window.should_close:
                    window.close()
                    continue
                window.update(dt)
                window.render()
                alive.append(window)
            self.windows = alive
            glfw.poll_events()
            self.update_fps(dt)
            
        for window in self.windows:
            window.close()
        glfw.terminate()
        self._running = False
