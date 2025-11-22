"""
Visualization package providing a minimal rendering stack with pluggable backends.

The module exposes abstractions for window/context management, scene graphs,
camera models and GPU resources such as meshes, shaders, materials and textures.
"""

from .window import Window, GLWindow
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
from .backends.base import GraphicsBackend, WindowBackend, MouseButton, Key, Action
from .backends.opengl import OpenGLGraphicsBackend
from .backends.glfw import GLFWWindowBackend
from .backends.qt import QtWindowBackend, QtGLWindowHandle

__all__ = [
    "Window",
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
    "GraphicsBackend",
    "WindowBackend",
    "MouseButton",
    "Key",
    "Action",
    "OpenGLGraphicsBackend",
    "GLFWWindowBackend",
    "QtWindowBackend",
    "QtGLWindowHandle",
]
