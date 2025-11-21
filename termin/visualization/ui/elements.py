from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import OpenGL.GL as gl

IDENTITY = np.identity(4, dtype=np.float32)

from ..material import Material


class UIElement:
    """Base UI element rendered inside a canvas."""

    material: Material | None = None

    def draw(self, canvas, context_key: int, viewport_rect: Tuple[int, int, int, int]):
        raise NotImplementedError

    def _require_material(self) -> Material:
        if self.material is None:
            raise RuntimeError(f"{self.__class__.__name__} has no material assigned.")
        return self.material


@dataclass
class UIRectangle(UIElement):
    """Axis-aligned rectangle defined in normalized viewport coordinates."""

    position: Tuple[float, float]
    size: Tuple[float, float]
    color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    material: Material | None = None

    def _to_clip_vertices(self) -> np.ndarray:
        x, y = self.position
        w, h = self.size
        left = x * 2.0 - 1.0
        right = (x + w) * 2.0 - 1.0
        top = 1.0 - y * 2.0
        bottom = 1.0 - (y + h) * 2.0
        return np.array(
            [
                [left, top],
                [right, top],
                [left, bottom],
                [right, bottom],
            ],
            dtype=np.float32,
        )

    def draw(self, canvas, context_key: int, viewport_rect: Tuple[int, int, int, int]):
        material = self._require_material()
        material.shader.ensure_ready()
        material.apply(IDENTITY, IDENTITY, IDENTITY)
        vertices = self._to_clip_vertices()
        shader = material.shader
        shader.set_uniform_vec4("u_color", np.array(self.color, dtype=np.float32))
        shader.set_uniform_int("u_use_texture", 0)
        canvas.draw_vertices(context_key, vertices)


@dataclass
class UIText(UIElement):
    text: str
    position: tuple[float, float]
    color: tuple[float, float, float, float] = (1, 1, 1, 1)
    scale: float = 1.0
    material: Material | None = None

    def draw(self, canvas, context_key, viewport):
        if not hasattr(canvas, "font"):
            return
        material = self._require_material()
        material.shader.ensure_ready()
        material.apply(IDENTITY, IDENTITY, IDENTITY)

        shader = material.shader
        shader.set_uniform_vec4("u_color", np.array(self.color, dtype=np.float32))
        shader.set_uniform_int("u_use_texture", 1)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, canvas.font.texture)
        shader.set_uniform_int("u_texture", 0)

        x, y = self.position
        px, py, pw, ph = viewport

        cx = x
        cy = y

        for ch in self.text:
            if ch not in canvas.font.glyphs:
                continue
            glyph = canvas.font.glyphs[ch]
            w, h = glyph["size"]
            u0, v0, u1, v1 = glyph["uv"]

            sx = w * self.scale / pw * 2
            sy = h * self.scale / ph * 2

            vx0 = cx * 2 - 1
            vy0 = 1 - cy * 2
            vx1 = (cx + w * self.scale / pw) * 2 - 1
            vy1 = 1 - (cy + h * self.scale / ph) * 2

            vertices = np.array([
                [vx0, vy0, u0, v0],
                [vx1, vy0, u1, v0],
                [vx0, vy1, u0, v1],
                [vx1, vy1, u1, v1],
            ], dtype=np.float32)

            canvas.draw_textured_quad(context_key, vertices)

            cx += (w * self.scale) / pw
