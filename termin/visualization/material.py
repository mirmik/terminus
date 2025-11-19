"""Material keeps shader reference and static uniform parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable

import numpy as np

from .shader import ShaderProgram
from .texture import Texture


class Material:
    """Collection of shader parameters applied before drawing a mesh."""

    def __init__(
        self,
        shader: ShaderProgram,
        color: np.ndarray | None = None,
        textures: Dict[str, Texture] | None = None,
        uniforms: Dict[str, Any] | None = None
    ):
        if color is None:
            color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        self.shader = shader
        self.color = color
        self.textures = textures if textures is not None else {}
        self.uniforms = uniforms if uniforms is not None else {}


    def apply(self, model: np.ndarray, view: np.ndarray, projection: np.ndarray):
        """Bind shader, upload MVP matrices and all statically defined uniforms."""
        self.shader.use()
        self.shader.set_uniform_matrix4("u_model", model)
        self.shader.set_uniform_matrix4("u_view", view)
        self.shader.set_uniform_matrix4("u_projection", projection)
        if self.color is not None:
            self.shader.set_uniform_vec3("u_color", self.color)

        texture_slots = enumerate(self.textures.items())
        for unit, (uniform_name, texture) in texture_slots:
            #print(f"Binding texture to unit {unit} for uniform {uniform_name}")
            texture.bind(unit)
            self.shader.set_uniform_int(uniform_name, unit)

        for name, value in self.uniforms.items():
            self.shader.set_uniform_auto(name, value)

    def update_color(self, rgba: Iterable[float]):
        vec = np.asarray(rgba, dtype=np.float32)
        if vec.shape != (4,):
            raise ValueError("Color must be an RGBA quadruplet.")
        self.color = vec
