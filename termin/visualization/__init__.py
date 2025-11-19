"""
Visualization package providing a minimal OpenGL-based rendering stack.

The module exposes abstractions for window/context management, scene graphs,
camera models and GPU resources such as meshes, shaders, materials and textures.
"""

from .window import GLWindow
from .renderer import Renderer
from .scene import Scene
from .entity import Entity
from .camera import Camera, PerspectiveCamera, OrthographicCamera, OrbitCamera
from .mesh import MeshDrawable
from .material import Material
from .shader import ShaderProgram
from .texture import Texture

__all__ = [
    "GLWindow",
    "Renderer",
    "Scene",
    "Entity",
    "Camera",
    "PerspectiveCamera",
    "OrthographicCamera",
    "OrbitCamera",
    "MeshDrawable",
    "Material",
    "ShaderProgram",
    "Texture",
]
