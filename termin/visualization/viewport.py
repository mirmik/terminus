
from dataclasses import dataclass, field
from typing import Optional, Tuple
from .scene import Scene
from .camera import CameraComponent

@dataclass
class Viewport:
    scene: Scene
    camera: CameraComponent
    rect: Tuple[float, float, float, float]
    canvas: Optional["Canvas"] = None
    postprocess: list["PostProcessEffect"] = field(default_factory=list)
