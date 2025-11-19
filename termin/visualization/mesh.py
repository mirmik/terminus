"""GPU mesh helper built on top of :mod:`termin.mesh` geometry."""

from __future__ import annotations

from typing import Optional

import numpy as np
from OpenGL import GL as gl
from OpenGL.raw.GL.VERSION.GL_2_0 import glVertexAttribPointer as _gl_vertex_attrib_pointer
import ctypes

from termin.mesh.mesh import Mesh


def _vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals: ``n_v = sum_{t∈F(v)} ( (v1-v0) × (v2-v0) ).``"""
    normals = np.zeros_like(vertices, dtype=np.float64)
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    for face, normal in zip(triangles, face_normals):
        normals[face] += normal
    norms = np.linalg.norm(normals, axis=1)
    norms[norms == 0] = 1.0
    return (normals.T / norms).T.astype(np.float32)


class MeshDrawable:
    """Uploads CPU mesh data to GPU buffers and issues draw commands."""

    def __init__(self, mesh: Mesh):
        self._mesh = mesh
        if self._mesh.vertex_normals is None:
            self._mesh.compute_vertex_normals()
        self._vao = None
        self._vbo = None
        self._ebo = None


    def upload(self):
        if self._vao is not None:
            return
        vertex_block = np.hstack((
            self._mesh.vertices, 
            self._mesh.vertex_normals,
            self._mesh.uv if self._mesh.uv is not None else np.zeros((self._mesh.vertices.shape[0], 2))    
        )).astype(np.float32).ravel()
        indices = self._mesh.triangles.astype(np.uint32).ravel()

        self._vao = gl.glGenVertexArrays(1)
        self._vbo = gl.glGenBuffers(1)
        self._ebo = gl.glGenBuffers(1)

        gl.glBindVertexArray(self._vao)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex_block.nbytes, vertex_block, gl.GL_STATIC_DRAW)

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)

        stride = 8 * 4  # 8 floats per vertex (3 position, 3 normal, 2 uv)
        gl.glEnableVertexAttribArray(0)
        _gl_vertex_attrib_pointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        _gl_vertex_attrib_pointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(12))
        gl.glEnableVertexAttribArray(2)
        _gl_vertex_attrib_pointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(24))

        gl.glBindVertexArray(0)

    def draw(self):
        if self._vao is None:
            self.upload()
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glBindVertexArray(self._vao)
        gl.glDrawElements(gl.GL_TRIANGLES, self._mesh.triangles.size, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))
        gl.glBindVertexArray(0)

    def delete(self):
        if self._vao is None:
            return
        gl.glDeleteVertexArrays(1, [self._vao])
        gl.glDeleteBuffers(1, [self._vbo])
        gl.glDeleteBuffers(1, [self._ebo])
        self._vao = self._vbo = self._ebo = None
