from __future__ import annotations

import ctypes
from typing import Dict, List, Tuple

from OpenGL import GL as gl
from OpenGL.raw.GL.VERSION.GL_2_0 import glVertexAttribPointer as _gl_vertex_attrib_pointer

from ..shader import ShaderProgram

UI_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec2 a_position;

void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

UI_FRAGMENT_SHADER = """
#version 330 core
uniform vec4 u_color;
out vec4 FragColor;

void main() {
    FragColor = u_color;
}
"""


class Canvas:
    """2D overlay composed of UI elements rendered in viewport space."""

    def __init__(self):
        self.elements: List["UIElement"] = []
        self.shader = ShaderProgram(UI_VERTEX_SHADER, UI_FRAGMENT_SHADER)
        self._buffers: Dict[int, Tuple[int, int]] = {}

    def add(self, element: "UIElement") -> "UIElement":
        self.elements.append(element)
        return element

    def remove(self, element: "UIElement"):
        if element in self.elements:
            self.elements.remove(element)

    def clear(self):
        self.elements.clear()

    def render(self, context_key: int, viewport_rect: Tuple[int, int, int, int]):
        if not self.elements:
            return
        self.shader.ensure_ready()
        self.shader.use()
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        for element in self.elements:
            element.draw(self, context_key, viewport_rect)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)
        self.shader.stop()

    def draw_vertices(self, context_key: int, vertices):
        vao, vbo = self._buffers.get(context_key, (None, None))
        if vao is None:
            vao = gl.glGenVertexArrays(1)
            vbo = gl.glGenBuffers(1)
            self._buffers[context_key] = (vao, vbo)
        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_DYNAMIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        _gl_vertex_attrib_pointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, ctypes.c_void_p(0))
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
        gl.glBindVertexArray(0)
