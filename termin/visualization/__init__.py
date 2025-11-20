"""
Visualization package providing a minimal OpenGL-based rendering stack.

The module exposes abstractions for window/context management, scene graphs,
camera models and GPU resources such as meshes, shaders, materials and textures.
"""

from .window import GLWindow
from .renderer import Renderer
from .scene import Scene
from .entity import Entity, Component, InputComponent, RenderContext
from .camera import (
    CameraComponent,
    PerspectiveCameraComponent,
    OrthographicCameraComponent,
    OrbitCameraController,
)
from .mesh import MeshDrawable
from .material import Material
from .shader import ShaderProgram
from .texture import Texture
from .components import MeshRenderer
from .ui import Canvas, UIElement, UIRectangle
from .world import VisualizationWorld

__all__ = [
    "GLWindow",
    "Renderer",
    "Scene",
    "Entity",
    "Component",
    "InputComponent",
    "RenderContext",
    "CameraComponent",
    "PerspectiveCameraComponent",
    "OrthographicCameraComponent",
    "OrbitCameraController",
    "MeshDrawable",
    "MeshRenderer",
    "Canvas",
    "UIElement",
    "UIRectangle",
    "Material",
    "ShaderProgram",
    "Texture",
    "VisualizationWorld",
]
