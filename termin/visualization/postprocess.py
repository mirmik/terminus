from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from .shader import ShaderProgram
from .backends.base import GraphicsBackend, FramebufferHandle, TextureHandle


@dataclass
class PostProcessEffect:
    """
    Один шаг постобработки: рендер сцены в текстуру + fullscreen quad.
    """
    shader: ShaderProgram
    fbos: Dict[int, FramebufferHandle] = field(default_factory=dict)

    def acquire_framebuffer(
        self,
        graphics: GraphicsBackend,
        context_key: int,
        size: Tuple[int, int],
    ) -> FramebufferHandle:
        fb = self.fbos.get(context_key)
        if fb is None:
            fb = graphics.create_framebuffer(size)
            self.fbos[context_key] = fb
        else:
            fb.resize(size)
        return fb

    def draw(self, graphics: GraphicsBackend, context_key: int, color_tex: TextureHandle):
        """
        Рисуем fullscreen quad, используя color_tex как источник.
        """
        self.shader.ensure_ready(graphics)
        self.shader.use()

        color_tex.bind(0)
        self.shader.set_uniform_int("u_texture", 0)

        # TRIANGLE_STRIP: 4 вершины, [-1..1] в клип-пространстве
        vertices = np.array(
            [
                [-1.0, -1.0, 0.0, 0.0],
                [ 1.0, -1.0, 1.0, 0.0],
                [-1.0,  1.0, 0.0, 1.0],
                [ 1.0,  1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )

        # Простое состояние: без depth, без blend
        graphics.set_depth_test(False)
        graphics.set_cull_face(False)
        graphics.set_blend(False)

        graphics.draw_ui_textured_quad(context_key, vertices)

        # Восстанавливать состояние глобально не будем — дальше всё равно UI.
