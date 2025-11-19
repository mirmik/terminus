"""Renderer configures OpenGL state and draws entities."""

from __future__ import annotations

import numpy as np
from OpenGL import GL as gl

from .camera import Camera, PerspectiveCamera
from .material import Material
from .scene import Scene
from .shader import ShaderProgram

from .opengl_helpers import init_opengl, opengl_is_inited


DEFAULT_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

out vec3 v_normal;
out vec3 v_world_pos;

void main() {
    vec4 world = u_model * vec4(a_position, 1.0);
    v_world_pos = world.xyz;
    v_normal = mat3(transpose(inverse(u_model))) * a_normal;
    gl_Position = u_projection * u_view * world;
}
"""


DEFAULT_FRAGMENT_SHADER = """
#version 330 core
in vec3 v_normal;
in vec3 v_world_pos;

uniform vec3 u_color;
uniform vec3 u_light_dir;

out vec4 FragColor;

void main() {
    vec3 normal = normalize(v_normal);
    float ndotl = max(dot(normal, -normalize(u_light_dir)), 0.0);
    vec3 diffuse = u_color * (0.3 + 0.7 * ndotl);
    FragColor = vec4(diffuse, 1.0);
}
"""



class Renderer:
    """Responsible for viewport setup, uniforms and draw traversal."""

    def __init__(self):
        pass

    def ensure_ready(self):
        if not opengl_is_inited():
            init_opengl()

    def _ensure_gl_state(self):
        if opengl_is_inited() == False:
            init_opengl()

    def render(self, scene: Scene, camera: Camera):
        self.ensure_ready()
        self._ensure_gl_state()
        bg = scene.background_color
        gl.glClearColor(float(bg[0]), float(bg[1]), float(bg[2]), float(bg[3]))
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        view = camera.get_view_matrix()
        projection = camera.get_projection_matrix()

        for entity in scene.entities:
            if not entity.visible:
                continue
            material = entity.material
            model = entity.model_matrix()
            material.apply(model, view, projection)
            material.shader.set_uniform_vec3("u_light_dir", scene.light_direction)
            material.shader.set_uniform_vec3("u_light_color", np.array([1.0, 1.0, 1.0], dtype=np.float32))
            entity.draw()

    def resize(self, width: int, height: int, camera: Camera):
        gl.glViewport(0, 0, width, height)
        if isinstance(camera, PerspectiveCamera):
            camera.set_aspect(width / max(1.0, float(height)))
