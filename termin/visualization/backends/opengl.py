"""OpenGL-based graphics backend."""

from __future__ import annotations

import ctypes
from typing import Dict, Tuple

import numpy as np
from OpenGL import GL as gl
from OpenGL.raw.GL.VERSION.GL_2_0 import glVertexAttribPointer as _gl_vertex_attrib_pointer

from termin.mesh.mesh import Mesh

from .base import (
    GraphicsBackend,
    MeshHandle,
    PolylineHandle,
    ShaderHandle,
    TextureHandle,
)

_OPENGL_INITED = False


def _compile_shader(source: str, shader_type: int) -> int:
    shader = gl.glCreateShader(shader_type)
    gl.glShaderSource(shader, source)
    gl.glCompileShader(shader)
    status = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)
    if not status:
        log = gl.glGetShaderInfoLog(shader)
        raise RuntimeError(log.decode("utf-8") if isinstance(log, bytes) else str(log))
    return shader


def _link_program(vertex_shader: int, fragment_shader: int) -> int:
    program = gl.glCreateProgram()
    gl.glAttachShader(program, vertex_shader)
    gl.glAttachShader(program, fragment_shader)
    gl.glLinkProgram(program)
    status = gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
    if not status:
        log = gl.glGetProgramInfoLog(program)
        raise RuntimeError(log.decode("utf-8") if isinstance(log, bytes) else str(log))
    gl.glDetachShader(program, vertex_shader)
    gl.glDetachShader(program, fragment_shader)
    gl.glDeleteShader(vertex_shader)
    gl.glDeleteShader(fragment_shader)
    return program


class OpenGLShaderHandle(ShaderHandle):
    def __init__(self, vertex_source: str, fragment_source: str):
        self.vertex_source = vertex_source
        self.fragment_source = fragment_source
        self.program: int | None = None
        self._uniform_cache: Dict[str, int] = {}

    def _ensure_compiled(self):
        if self.program is not None:
            return
        vert = _compile_shader(self.vertex_source, gl.GL_VERTEX_SHADER)
        frag = _compile_shader(self.fragment_source, gl.GL_FRAGMENT_SHADER)
        self.program = _link_program(vert, frag)

    def use(self):
        self._ensure_compiled()
        gl.glUseProgram(self.program)

    def stop(self):
        gl.glUseProgram(0)

    def delete(self):
        if self.program is not None:
            gl.glDeleteProgram(self.program)
            self.program = None
        self._uniform_cache.clear()

    def _uniform_location(self, name: str) -> int:
        if name not in self._uniform_cache:
            location = gl.glGetUniformLocation(self.program, name.encode("utf-8"))
            self._uniform_cache[name] = location
        return self._uniform_cache[name]

    def set_uniform_matrix4(self, name: str, matrix):
        self._ensure_compiled()
        mat = np.asarray(matrix, dtype=np.float32)
        gl.glUniformMatrix4fv(self._uniform_location(name), 1, True, mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    def set_uniform_vec3(self, name: str, vector):
        self._ensure_compiled()
        vec = np.asarray(vector, dtype=np.float32)
        gl.glUniform3f(self._uniform_location(name), float(vec[0]), float(vec[1]), float(vec[2]))

    def set_uniform_vec4(self, name: str, vector):
        self._ensure_compiled()
        vec = np.asarray(vector, dtype=np.float32)
        gl.glUniform4f(self._uniform_location(name), float(vec[0]), float(vec[1]), float(vec[2]), float(vec[3]))

    def set_uniform_float(self, name: str, value: float):
        self._ensure_compiled()
        gl.glUniform1f(self._uniform_location(name), float(value))

    def set_uniform_int(self, name: str, value: int):
        self._ensure_compiled()
        gl.glUniform1i(self._uniform_location(name), int(value))


class OpenGLMeshHandle(MeshHandle):
    def __init__(self, mesh: Mesh):
        self._mesh = mesh
        if self._mesh.vertex_normals is None:
            self._mesh.compute_vertex_normals()
        self._vao: int | None = None
        self._vbo: int | None = None
        self._ebo: int | None = None
        self._index_count = self._mesh.triangles.size
        self._upload()

    def _upload(self):
        vertex_block = np.hstack((
            self._mesh.vertices,
            self._mesh.vertex_normals,
            self._mesh.uv if self._mesh.uv is not None else np.zeros((self._mesh.vertices.shape[0], 2)),
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
        stride = 8 * 4
        gl.glEnableVertexAttribArray(0)
        _gl_vertex_attrib_pointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        _gl_vertex_attrib_pointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(12))
        gl.glEnableVertexAttribArray(2)
        _gl_vertex_attrib_pointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(24))
        gl.glBindVertexArray(0)

    def draw(self):
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glBindVertexArray(self._vao or 0)
        gl.glDrawElements(gl.GL_TRIANGLES, self._index_count, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))
        gl.glBindVertexArray(0)

    def delete(self):
        if self._vao is None:
            return
        gl.glDeleteVertexArrays(1, [self._vao])
        gl.glDeleteBuffers(1, [self._vbo])
        gl.glDeleteBuffers(1, [self._ebo])
        self._vao = self._vbo = self._ebo = None


class OpenGLPolylineHandle(PolylineHandle):
    def __init__(self, vertices: np.ndarray, indices: np.ndarray | None, is_strip: bool):
        self._vertices = vertices.astype(np.float32)
        self._indices = indices.astype(np.uint32) if indices is not None else None
        self._is_strip = is_strip
        self._vao: int | None = None
        self._vbo: int | None = None
        self._ebo: int | None = None
        self._upload()

    def _upload(self):
        vertex_block = self._vertices.ravel()
        self._vao = gl.glGenVertexArrays(1)
        self._vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self._vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex_block.nbytes, vertex_block, gl.GL_STATIC_DRAW)
        if self._indices is not None:
            self._ebo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, self._indices.nbytes, self._indices, gl.GL_STATIC_DRAW)
        stride = 3 * 4
        gl.glEnableVertexAttribArray(0)
        _gl_vertex_attrib_pointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
        gl.glBindVertexArray(0)

    def draw(self):
        mode = gl.GL_LINE_STRIP if self._is_strip else gl.GL_LINES
        gl.glBindVertexArray(self._vao or 0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        if self._indices is not None:
            gl.glDrawElements(mode, self._indices.size, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))
        else:
            gl.glDrawArrays(mode, 0, self._vertices.shape[0])
        gl.glBindVertexArray(0)

    def delete(self):
        if self._vao is None:
            return
        gl.glDeleteVertexArrays(1, [self._vao])
        gl.glDeleteBuffers(1, [self._vbo])
        if self._ebo is not None:
            gl.glDeleteBuffers(1, [self._ebo])
        self._vao = self._vbo = self._ebo = None


class OpenGLTextureHandle(TextureHandle):
    def __init__(self, image_data: np.ndarray, size: Tuple[int, int], channels: int = 4, mipmap: bool = True, clamp: bool = False):
        self._handle: int | None = None
        self._channels = channels
        self._data = image_data
        self._size = size
        self._mipmap = mipmap
        self._clamp = clamp
        self._upload()

    def _upload(self):
        self._handle = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._handle)
        internal_format = gl.GL_RGBA if self._channels != 1 else gl.GL_RED
        gl_format = internal_format
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal_format, self._size[0], self._size[1], 0, gl_format, gl.GL_UNSIGNED_BYTE, self._data)
        if self._mipmap:
            gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        min_filter = gl.GL_LINEAR_MIPMAP_LINEAR if self._mipmap else gl.GL_LINEAR
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, min_filter)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        wrap_mode = gl.GL_CLAMP_TO_EDGE if self._clamp else gl.GL_REPEAT
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, wrap_mode)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, wrap_mode)
        if self._channels == 1:
            swizzle = np.array([gl.GL_RED, gl.GL_RED, gl.GL_RED, gl.GL_RED], dtype=np.int32)
            gl.glTexParameteriv(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_SWIZZLE_RGBA, swizzle)

    def bind(self, unit: int = 0):
        gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._handle or 0)

    def delete(self):
        if self._handle is not None:
            gl.glDeleteTextures(1, [self._handle])
            self._handle = None


class OpenGLGraphicsBackend(GraphicsBackend):
    def __init__(self):
        self._ui_buffers: Dict[int, Tuple[int, int]] = {}

    def ensure_ready(self):
        print("ENSURE READY")
        global _OPENGL_INITED
        if _OPENGL_INITED:
            return
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)
        gl.glFrontFace(gl.GL_CCW)
        _OPENGL_INITED = True

    def set_viewport(self, x: int, y: int, w: int, h: int):
        gl.glViewport(x, y, w, h)

    def enable_scissor(self, x: int, y: int, w: int, h: int):
        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(x, y, w, h)

    def disable_scissor(self):
        gl.glDisable(gl.GL_SCISSOR_TEST)

    def clear_color_depth(self, color):
        gl.glClearColor(float(color[0]), float(color[1]), float(color[2]), float(color[3]))
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    def set_depth_test(self, enabled: bool):
        if enabled:
            gl.glEnable(gl.GL_DEPTH_TEST)
        else:
            gl.glDisable(gl.GL_DEPTH_TEST)

    def set_depth_mask(self, enabled: bool):
        gl.glDepthMask(gl.GL_TRUE if enabled else gl.GL_FALSE)

    def set_depth_func(self, func: str):
        mapping = {"less": gl.GL_LESS, "lequal": gl.GL_LEQUAL}
        gl.glDepthFunc(mapping.get(func, gl.GL_LESS))

    def set_cull_face(self, enabled: bool):
        if enabled:
            gl.glEnable(gl.GL_CULL_FACE)
        else:
            gl.glDisable(gl.GL_CULL_FACE)

    def set_blend(self, enabled: bool):
        if enabled:
            gl.glEnable(gl.GL_BLEND)
        else:
            gl.glDisable(gl.GL_BLEND)

    def set_blend_func(self, src: str, dst: str):
        mapping = {
            "src_alpha": gl.GL_SRC_ALPHA,
            "one_minus_src_alpha": gl.GL_ONE_MINUS_SRC_ALPHA,
            "one": gl.GL_ONE,
            "zero": gl.GL_ZERO,
        }
        gl.glBlendFunc(mapping.get(src, gl.GL_SRC_ALPHA), mapping.get(dst, gl.GL_ONE_MINUS_SRC_ALPHA))

    def create_shader(self, vertex_source: str, fragment_source: str) -> ShaderHandle:
        return OpenGLShaderHandle(vertex_source, fragment_source)

    def create_mesh(self, mesh: Mesh) -> MeshHandle:
        return OpenGLMeshHandle(mesh)

    def create_polyline(self, polyline) -> PolylineHandle:
        return OpenGLPolylineHandle(polyline.vertices, polyline.indices, polyline.is_strip)

    def create_texture(self, image_data, size: Tuple[int, int], channels: int = 4, mipmap: bool = True, clamp: bool = False) -> TextureHandle:
        return OpenGLTextureHandle(image_data, size, channels=channels, mipmap=mipmap, clamp=clamp)

    def draw_ui_vertices(self, context_key: int, vertices):
        vao, vbo = self._ui_buffers.get(context_key, (None, None))
        if vao is None:
            vao = gl.glGenVertexArrays(1)
            vbo = gl.glGenBuffers(1)
            self._ui_buffers[context_key] = (vao, vbo)
        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_DYNAMIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        _gl_vertex_attrib_pointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, ctypes.c_void_p(0))
        gl.glDisableVertexAttribArray(1)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
        gl.glBindVertexArray(0)

    def draw_ui_textured_quad(self, context_key: int, vertices):
        vao, vbo = self._ui_buffers.get(context_key, (None, None))
        if vao is None:
            vao = gl.glGenVertexArrays(1)
            vbo = gl.glGenBuffers(1)
            self._ui_buffers[context_key] = (vao, vbo)
        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_DYNAMIC_DRAW)
        stride = 4 * 4
        gl.glEnableVertexAttribArray(0)
        _gl_vertex_attrib_pointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        _gl_vertex_attrib_pointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(8))
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
        gl.glBindVertexArray(0)
