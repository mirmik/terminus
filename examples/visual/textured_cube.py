"""Minimal demo that renders a cube and allows orbiting camera controls."""

from __future__ import annotations

import numpy as np

from termin.geombase.pose3 import Pose3
from termin.mesh.mesh import Mesh, CubeMesh, UVSphereMesh, IcoSphereMesh, PlaneMesh, CylinderMesh, ConeMesh, TexturedCubeMesh
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

    // нормальная матрица = mat3(transpose(inverse(u_model)))
    v_normal = mat3(transpose(inverse(u_model))) * a_normal;

    gl_Position = u_projection * u_view * world;

    // Простейшее текстурирование по проекции куба
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
uniform sampler2D u_diffuse_map;

out vec4 FragColor;

void main() {
    // Нормаль
    vec3 N = normalize(v_normal);

    // Направление на свет: если u_light_dir - направление *от* света, то на объект оно то же
    vec3 L = normalize(-u_light_dir); // если задаёшь уже "к объекту", убери минус

    // Направление на камеру
    vec3 V = normalize(u_view_pos - v_world_pos);

    // Полуунитектор (half-vector) для Blinn–Phong
    vec3 H = normalize(L + V);

    // --- коэффициенты освещения ---
    const float ambientStrength  = 0.2;  // эмбиент
    const float diffuseStrength  = 0.8;  // диффуз
    const float specularStrength = 0.4;  // спекуляр
    const float shininess        = 32.0; // степень блеска

    // Эмбиент
    vec3 ambient = ambientStrength * u_color;

    // Диффуз (Ламберт)
    float ndotl = max(dot(N, L), 0.0);
    vec3 texColor = texture(u_diffuse_map, v_texcoord).rgb;
    vec3 diffuse = diffuseStrength * ndotl * texColor;
    diffuse = texColor;

    // Спекуляр (Blinn–Phong)
    float specFactor = 0.0;
    if (ndotl > 0.0) {
        specFactor = pow(max(dot(N, H), 0.0), shininess);
    }
    vec3 specular = specularStrength * specFactor * u_light_color;

    // Итоговый цвет: модифицируем цвет материала цветом света
    //vec3 color = (ambient + diffuse) * u_light_color + specular;
    vec3 color = texColor;

    // Можно слегка ограничить, чтобы не улетало в дикий перегиб
    color = clamp(color, 0.0, 1.0);

    FragColor = vec4(color, 1.0);
}
"""

def build_scene() -> Scene:

    texture_path = "examples/data/textures/crate_diffuse.png"
    texture = Texture.from_file(texture_path)

    cube_mesh = TexturedCubeMesh()
    drawable = MeshDrawable(cube_mesh)
    shader_prog = ShaderProgram(vert, frag)
    renderer = Renderer()
    material = Material(shader=shader_prog, color=np.array([0.8, 0.3, 0.3, 1.0], dtype=np.float32), textures = {"u_diffuse_map": texture})
    entity = Entity(mesh=drawable, material=material, pose=Pose3.identity(), name="cube")
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
