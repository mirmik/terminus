"""OpenGL shader helpers implemented with PyOpenGL + GLFW contexts."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .opengl_helpers import init_opengl, opengl_is_inited

import numpy as np
from OpenGL import GL as gl

import sys


class ShaderCompilationError(RuntimeError):
    """Raised when GLSL compilation or program linking fails."""


def _compile_shader(source: str, shader_type: int) -> int:
    shader = gl.glCreateShader(shader_type)
    gl.glShaderSource(shader, source)
    gl.glCompileShader(shader)

    status = ctypes.c_int()
    gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS, ctypes.byref(status))
    if not status.value:
        log = gl.glGetShaderInfoLog(shader)
        sys.stderr.write(f"Shader compilation failed:\n{log}\n")
        raise ShaderCompilationError(log.decode("utf-8") if isinstance(log, bytes) else str(log))
    return shader


def _link_program(vertex_shader: int, fragment_shader: int) -> int:
    program = gl.glCreateProgram()
    gl.glAttachShader(program, vertex_shader)
    gl.glAttachShader(program, fragment_shader)
    gl.glLinkProgram(program)

    status = ctypes.c_int()
    gl.glGetProgramiv(program, gl.GL_LINK_STATUS, ctypes.byref(status))
    if not status.value:
        log = gl.glGetProgramInfoLog(program)
        raise ShaderCompilationError(log.decode("utf-8") if isinstance(log, bytes) else str(log))
    gl.glDetachShader(program, vertex_shader)
    gl.glDetachShader(program, fragment_shader)
    gl.glDeleteShader(vertex_shader)
    gl.glDeleteShader(fragment_shader)
    return program


class ShaderProgram:
    """A GLSL shader program (vertex + fragment).

    Uniform setters inside the class assume column-major matrices and they set the
    combined MVP transform ``P * V * M`` in homogeneous coordinates.
    """

    def __init__(
        self,
        vertex_source: str,
        fragment_source: str
    ):
        self.vertex_source = vertex_source
        self.fragment_source = fragment_source
        self.program: Optional[int] = None
        self._uniform_cache: Dict[str, int] = {}
        self._compiled = False

    def __post_init__(self):
        self._uniform_cache = {}

    def compile(self):
        vertex_shader = _compile_shader(self.vertex_source, gl.GL_VERTEX_SHADER)
        fragment_shader = _compile_shader(self.fragment_source, gl.GL_FRAGMENT_SHADER)
        self.program = _link_program(vertex_shader, fragment_shader)
        self._compiled = True

    def ensure_ready(self):
        if not self._compiled:
            self.compile()

    def use(self):
        gl.glUseProgram(self.program)

    def stop(self):
        gl.glUseProgram(0)

    def delete(self):
        if self.program:
            gl.glDeleteProgram(self.program)
            self.program = None

    def uniform_location(self, name: str) -> int:
        if name not in self._uniform_cache:
            location = gl.glGetUniformLocation(self.program, name.encode("utf-8"))
            self._uniform_cache[name] = location
        return self._uniform_cache[name]

    def set_uniform_matrix4(self, name: str, matrix: np.ndarray):
        """Upload a 4x4 matrix (float32) to uniform ``name``."""
        location = self.uniform_location(name)
        mat = np.asarray(matrix, dtype=np.float32)
        gl.glUniformMatrix4fv(location, 1, True, mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    def set_uniform_vec3(self, name: str, vector: np.ndarray):
        location = self.uniform_location(name)
        vec = np.asarray(vector, dtype=np.float32)
        gl.glUniform3f(location, float(vec[0]), float(vec[1]), float(vec[2]))

    def set_uniform_vec4(self, name: str, vector: np.ndarray):
        location = self.uniform_location(name)
        vec = np.asarray(vector, dtype=np.float32)
        gl.glUniform4f(location, float(vec[0]), float(vec[1]), float(vec[2]), float(vec[3]))

    def set_uniform_float(self, name: str, value: float):
        location = self.uniform_location(name)
        gl.glUniform1f(location, float(value))

    def set_uniform_int(self, name: str, value: int):
        location = self.uniform_location(name)
        gl.glUniform1i(location, int(value))

    def set_uniform_auto(self, name: str, value: Any):
        """Best-effort setter that infers uniform type based on ``value``."""
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value)
            if arr.shape == (4, 4):
                self.set_uniform_matrix4(name, arr)
            elif arr.shape == (3,):
                self.set_uniform_vec3(name, arr)
            elif arr.shape == (4,):
                self.set_uniform_vec4(name, arr)
            else:
                raise ValueError(f"Unsupported uniform array shape for {name}: {arr.shape}")
        elif isinstance(value, bool):
            self.set_uniform_int(name, int(value))
        elif isinstance(value, int):
            self.set_uniform_int(name, value)
        else:
            self.set_uniform_float(name, float(value))

    @classmethod
    def from_files(cls, vertex_path: str | Path, fragment_path: str | Path) -> "ShaderProgram":
        vertex_source = Path(vertex_path).read_text(encoding="utf-8")
        fragment_source = Path(fragment_path).read_text(encoding="utf-8")
        return cls(vertex_source=vertex_source, fragment_source=fragment_source)
