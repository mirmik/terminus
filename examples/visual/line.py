"""Minimal demo that renders a cube and allows orbiting camera controls."""

from __future__ import annotations

import numpy as np

from termin.geombase.pose3 import Pose3
from termin.mesh.mesh import Mesh, CubeMesh, UVSphereMesh, IcoSphereMesh, PlaneMesh, CylinderMesh, ConeMesh, TexturedCubeMesh
from termin.visualization.line import LineEntity
from termin.visualization import Entity, GLWindow, MeshDrawable, Renderer, Scene, Material, Texture
from termin.visualization.shader import ShaderProgram
from termin.visualization.skybox import SkyBoxEntity

vert = """
#version 330 core
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_texcoord;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

out vec3 v_normal;     // нормаль в мировом пространстве
out vec3 v_world_pos;  // позиция в мировом пространстве
out vec2 v_texcoord;

void main() {
    vec4 world = u_model * vec4(a_position, 1.0);
    v_world_pos = world.xyz;
    v_normal = mat3(transpose(inverse(u_model))) * a_normal;
    gl_Position = u_projection * u_view * world;
    v_texcoord = a_texcoord;    
}
"""


frag = """
#version 330 core

in vec3 v_normal;
in vec3 v_world_pos;
in vec2 v_texcoord;

uniform vec3 u_color;        // базовый цвет материала
uniform vec3 u_light_dir;    // направление от источника к объекту (world space)
uniform vec3 u_light_color;  // цвет света
uniform vec3 u_view_pos;     // позиция камеры (world space)

out vec4 FragColor;

void main() {
    vec3 col = u_color;
    FragColor = vec4(col, 1.0);
}
"""

def build_scene() -> Scene:

    texture_path = "examples/data/textures/crate_diffuse.png"
    texture = Texture.from_file(texture_path)
    shader_prog = ShaderProgram(vert, frag)
    material = Material(shader=shader_prog, color=np.array([0.8, 0.3, 0.3, 1.0], dtype=np.float32), textures = {"u_diffuse_map": texture})

    line = LineEntity(points=[
        np.array([0,0,0]), 
        np.array([1,0,0]),
        np.array([1,1,0]),
        np.array([0,1,0]),
        np.array([0,0,0])    
        ], color=np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32), width=2.0, material=material)
    #drawable = MeshDrawable(cube_mesh)
    renderer = Renderer()
    entity = Entity(mesh=line, material=material, pose=Pose3.identity(), name="cube")
    scene = Scene()
    scene.add(entity)

    skybox = SkyBoxEntity()
    scene.add(skybox)
    
    return scene, renderer


def main():
    scene, renderer = build_scene()
    window = GLWindow(scene=scene, renderer=renderer, title="termin cube demo")
    window.run()


if __name__ == "__main__":
    main()
