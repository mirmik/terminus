from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


class UIElement:
    """Base UI element rendered inside a canvas."""

    def draw(self, canvas, context_key: int, viewport_rect: Tuple[int, int, int, int]):
        raise NotImplementedError


@dataclass
class UIRectangle(UIElement):
    """Axis-aligned rectangle defined in normalized viewport coordinates."""

    position: Tuple[float, float]
    size: Tuple[float, float]
    color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

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
        vertices = self._to_clip_vertices()
        canvas.shader.set_uniform_vec4("u_color", np.array(self.color, dtype=np.float32))
        canvas.draw_vertices(context_key, vertices)
