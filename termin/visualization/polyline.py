from __future__ import annotations
from typing import Optional

import numpy as np
from OpenGL import GL as gl
from OpenGL.raw.GL.VERSION.GL_2_0 import glVertexAttribPointer as _gl_vertex_attrib_pointer
import ctypes


class Polyline:
    """
    Минимальная структура данных:
    vertices: (N, 3)
    indices: optional (M,) — индексы для линий; если None, рисуем по порядку
    is_strip: bool — GL_LINE_STRIP или GL_LINES
    """
    def __init__(self,
                 vertices: np.ndarray,
                 indices: Optional[np.ndarray] = None,
                 is_strip: bool = True):
        self.vertices = vertices.astype(np.float32)
        self.indices = indices.astype(np.uint32) if indices is not None else None
        self.is_strip = is_strip


class PolylineDrawable:
    """Рисует полилинию из CPU данных."""
    
    def __init__(self, polyline: Polyline):
        self._poly = polyline
        self._vao = None
        self._vbo = None
        self._ebo = None

    def upload(self):
        if self._vao is not None:
            return
        
        vertex_block = self._poly.vertices.ravel()
        
        self._vao = gl.glGenVertexArrays(1)
        self._vbo = gl.glGenBuffers(1)
        self._ebo = gl.glGenBuffers(1) if self._poly.indices is not None else None

        gl.glBindVertexArray(self._vao)

        # vertex buffer
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex_block.nbytes, vertex_block, gl.GL_STATIC_DRAW)

        # index buffer (optional)
        if self._poly.indices is not None:
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER,
                            self._poly.indices.nbytes,
                            self._poly.indices,
                            gl.GL_STATIC_DRAW)

        # vertex layout: only positions
        stride = 3 * 4
        gl.glEnableVertexAttribArray(0)
        _gl_vertex_attrib_pointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))

        gl.glBindVertexArray(0)

    def draw(self):
        if self._vao is None:
            self.upload()
        
        mode = gl.GL_LINE_STRIP if self._poly.is_strip else gl.GL_LINES
        
        gl.glBindVertexArray(self._vao)
        gl.glEnable(gl.GL_DEPTH_TEST)

        if self._poly.indices is not None:
            gl.glDrawElements(
                mode,
                self._poly.indices.size,
                gl.GL_UNSIGNED_INT,
                ctypes.c_void_p(0)
            )
        else:
            gl.glDrawArrays(
                mode,
                0,
                self._poly.vertices.shape[0]
            )

        gl.glBindVertexArray(0)

    def delete(self):
        if self._vao is None:
            return
        
        gl.glDeleteVertexArrays(1, [self._vao])
        gl.glDeleteBuffers(1, [self._vbo])
        if self._ebo is not None:
            gl.glDeleteBuffers(1, [self._ebo])
        self._vao = self._vbo = self._ebo = None