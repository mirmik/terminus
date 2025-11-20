from unittest import mock

import numpy as np
import pytest
from PIL import Image

import termin.visualization.mesh as mesh_module
import termin.visualization.shader as shader_module
import termin.visualization.texture as texture_module
from termin.visualization.mesh import MeshDrawable, _vertex_normals
from termin.visualization.shader import ShaderCompilationError, ShaderProgram
from termin.visualization.texture import Texture


class FakeShaderGL:
    GL_VERTEX_SHADER = 0x8B31
    GL_FRAGMENT_SHADER = 0x8B30
    GL_COMPILE_STATUS = 0x8B81
    GL_LINK_STATUS = 0x8B82

    def __init__(self, shader_status=None, shader_logs=None, program_status=None, program_logs=None):
        self.shader_status = list(shader_status or [])
        self.shader_logs = list(shader_logs or [])
        self.program_status = list(program_status or [])
        self.program_logs = list(program_logs or [])
        self.next_id = 1
        self.created_shaders = []
        self.created_programs = []

    def _next_status(self, queue):
        return queue.pop(0) if queue else 1

    def glCreateShader(self, shader_type):
        self.created_shaders.append(shader_type)
        shader_id = self.next_id
        self.next_id += 1
        return shader_id

    def glShaderSource(self, shader, source):
        return None

    def glCompileShader(self, shader):
        return None

    @staticmethod
    def _write_pointer(ptr, value):
        if hasattr(ptr, "contents"):
            ptr.contents.value = value
        else:
            ptr._obj.value = value

    def glGetShaderiv(self, shader, pname, params):
        self._write_pointer(params, self._next_status(self.shader_status))

    def glGetShaderInfoLog(self, shader):
        if self.shader_logs:
            return self.shader_logs.pop(0)
        return b""

    def glCreateProgram(self):
        program_id = self.next_id
        self.next_id += 1
        self.created_programs.append(program_id)
        return program_id

    def glAttachShader(self, program, shader):
        return None

    def glLinkProgram(self, program):
        return None

    def glGetProgramiv(self, program, pname, params):
        self._write_pointer(params, self._next_status(self.program_status))

    def glGetProgramInfoLog(self, program):
        if self.program_logs:
            return self.program_logs.pop(0)
        return b""

    def glDetachShader(self, program, shader):
        return None

    def glDeleteShader(self, shader):
        return None

    def glDeleteProgram(self, program):
        return None

    def glUseProgram(self, program):
        return None

    def glGetUniformLocation(self, program, name):
        return 0

    def glUniformMatrix4fv(self, location, count, transpose, value):
        return None

    def glUniform3f(self, location, x, y, z):
        return None

    def glUniform1f(self, location, value):
        return None

    def glUniform1i(self, location, value):
        return None


class FakeMeshGL:
    GL_ARRAY_BUFFER = 0x8892
    GL_ELEMENT_ARRAY_BUFFER = 0x8893
    GL_STATIC_DRAW = 0x88E4
    GL_FLOAT = 0x1406
    GL_FALSE = 0
    GL_TRIANGLES = 0x0004
    GL_UNSIGNED_INT = 0x1405
    GL_DEPTH_TEST = 0x0B71

    def __init__(self):
        self.next_id = 1
        self.bound_vao = None
        self.bound_vbo = None
        self.bound_ebo = None
        self.enabled = set()
        self.draw_calls = 0
        self.last_draw_count = 0
        self.array_buffer_size = 0
        self.element_buffer_size = 0

    def _new_id(self):
        result = self.next_id
        self.next_id += 1
        return result

    def glGenVertexArrays(self, count):
        return self._new_id()

    def glGenBuffers(self, count):
        return self._new_id()

    def glBindVertexArray(self, vao):
        self.bound_vao = vao

    def glBindBuffer(self, target, buffer_id):
        if target == self.GL_ARRAY_BUFFER:
            self.bound_vbo = buffer_id
        elif target == self.GL_ELEMENT_ARRAY_BUFFER:
            self.bound_ebo = buffer_id

    def glBufferData(self, target, size, data, usage):
        if target == self.GL_ARRAY_BUFFER:
            self.array_buffer_size = size
        elif target == self.GL_ELEMENT_ARRAY_BUFFER:
            self.element_buffer_size = size

    def glEnableVertexAttribArray(self, location):
        return None

    def glVertexAttribPointer(self, index, size, dtype, normalized, stride, pointer):
        return None

    def glEnable(self, cap):
        self.enabled.add(cap)

    def glDrawElements(self, mode, count, dtype, indices):
        self.draw_calls += 1
        self.last_draw_count = count

    def glDeleteVertexArrays(self, count, vaos):
        return None

    def glDeleteBuffers(self, count, buffers):
        return None


class FakeTextureGL:
    GL_TEXTURE_2D = 0x0DE1
    GL_TEXTURE0 = 0x84C0
    GL_RGBA = 0x1908
    GL_UNSIGNED_BYTE = 0x1401
    GL_LINEAR_MIPMAP_LINEAR = 0x2703
    GL_LINEAR = 0x2601
    GL_REPEAT = 0x2901
    GL_TEXTURE_MIN_FILTER = 0x2801
    GL_TEXTURE_MAG_FILTER = 0x2800
    GL_TEXTURE_WRAP_S = 0x2802
    GL_TEXTURE_WRAP_T = 0x2803

    def __init__(self):
        self.generated_ids = []
        self.bound_texture = None
        self.active_unit = None
        self.tex_image_calls = 0
        self.params = []
        self.mipmaps = 0

    def glGenTextures(self, count):
        tex_id = len(self.generated_ids) + 1
        self.generated_ids.append(tex_id)
        return tex_id

    def glBindTexture(self, target, tex_id):
        self.bound_texture = tex_id

    def glTexImage2D(self, *args, **kwargs):
        self.tex_image_calls += 1

    def glGenerateMipmap(self, target):
        self.mipmaps += 1

    def glTexParameteri(self, target, pname, param):
        self.params.append((pname, param))

    def glActiveTexture(self, unit):
        self.active_unit = unit


def test_vertex_normals_are_normalized():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    normals = _vertex_normals(vertices, triangles)
    lengths = np.linalg.norm(normals, axis=1)
    np.testing.assert_allclose(lengths, np.ones_like(lengths))
    np.testing.assert_allclose(normals[:, 2], np.ones(4))


def test_mesh_upload_and_draw_without_real_context():
    fake_gl = FakeMeshGL()
    pointer_calls = []

    def fake_pointer(index, size, dtype, normalized, stride, pointer):
        pointer_calls.append((index, stride))

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    triangles = np.array([[0, 1, 2]], dtype=int)
    mesh = mesh_module.Mesh(vertices, triangles)
    fake_ctx = 1234
    fake_glfw = mock.Mock()
    fake_glfw.get_current_context.return_value = fake_ctx
    with mock.patch.object(mesh_module, "gl", fake_gl), \
         mock.patch.object(mesh_module, "glfw", fake_glfw), \
         mock.patch.object(mesh_module, "_gl_vertex_attrib_pointer", side_effect=fake_pointer):
        drawable = MeshDrawable(mesh)
        drawable.draw(fake_ctx)
        assert fake_ctx in drawable._context_resources
        assert fake_gl.draw_calls == 1
        expected_array_bytes = mesh.vertices.shape[0] * 8 * 4
        assert fake_gl.array_buffer_size == expected_array_bytes
        assert fake_gl.element_buffer_size == triangles.size * 4
        assert pointer_calls == [(0, 32), (1, 32), (2, 32)]


def test_shader_compilation_success_with_fake_gl():
    fake_gl = FakeShaderGL()
    with mock.patch.object(shader_module, "gl", fake_gl):
        shader = ShaderProgram("void main(){ }", "void main(){ }")
        shader.compile()
        assert shader.program is not None
        assert len(fake_gl.created_shaders) == 2


def test_shader_compilation_failure_reports_log():
    fake_gl = FakeShaderGL(shader_status=[0], shader_logs=[b"bad shader"])
    with mock.patch.object(shader_module, "gl", fake_gl):
        shader = ShaderProgram("void main(){ }", "void main(){ }")
        with pytest.raises(ShaderCompilationError) as exc:
            shader.compile()
        assert "bad shader" in str(exc.value)


def test_texture_upload_defers_until_bind(tmp_path):
    fake_gl = FakeTextureGL()
    image_path = tmp_path / "tex.png"
    Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(image_path)
    with mock.patch.object(texture_module, "gl", fake_gl):
        tex = Texture.from_file(image_path)
        assert fake_gl.generated_ids == []
        tex.bind(2)
        assert fake_gl.generated_ids == [1]
        assert fake_gl.active_unit == fake_gl.GL_TEXTURE0 + 2
        assert fake_gl.bound_texture == 1
        assert fake_gl.tex_image_calls == 1
        tex.bind(2)
        assert fake_gl.tex_image_calls == 1  # No re-upload
