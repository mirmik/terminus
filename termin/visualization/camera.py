"""Camera classes producing view/projection matrices."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from termin.geombase.pose3 import Pose3


@dataclass
class Camera:
    """Base camera that stores pose plus clipping planes."""

    pose: Pose3
    near: float = 0.1
    far: float = 100.0

    def get_view_matrix(self) -> np.ndarray:
        """Calculate view matrix ``V = (pose)^{-1}``."""
        return self.pose.inverse().as_matrix()

    def get_projection_matrix(self) -> np.ndarray:
        raise NotImplementedError


class PerspectiveCamera(Camera):
    """Perspective projection defined by field-of-view and aspect ratio."""

    def __init__(self, pose: Pose3 | None = None, fov_y_degrees: float = 60.0, aspect: float = 1.0, near: float = 0.1, far: float = 100.0):
        super().__init__(pose=pose or Pose3.identity(), near=near, far=far)
        self.fov_y = math.radians(fov_y_degrees)
        self.aspect = aspect

    def set_aspect(self, aspect: float):
        self.aspect = aspect

    def get_projection_matrix(self) -> np.ndarray:
        """Standard perspective matrix with ``f = 1 / tan(fov/2)``."""
        f = 1.0 / math.tan(self.fov_y * 0.5)
        near, far = self.near, self.far
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / self.aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2 * far * near) / (near - far)
        proj[3, 2] = -1.0
        return proj


class OrthographicCamera(Camera):
    """Orthographic camera defined by view volume."""

    def __init__(self, pose: Pose3 | None = None, left: float = -1.0, right: float = 1.0, bottom: float = -1.0, top: float = 1.0, near: float = 0.1, far: float = 100.0):
        super().__init__(pose=pose or Pose3.identity(), near=near, far=far)
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top

    def get_projection_matrix(self) -> np.ndarray:
        lr = self.right - self.left
        tb = self.top - self.bottom
        fn = self.far - self.near
        proj = np.identity(4, dtype=np.float32)
        proj[0, 0] = 2.0 / lr
        proj[1, 1] = 2.0 / tb
        proj[2, 2] = -2.0 / fn
        proj[0, 3] = -(self.right + self.left) / lr
        proj[1, 3] = -(self.top + self.bottom) / tb
        proj[2, 3] = -(self.far + self.near) / fn
        return proj


class OrbitCamera(PerspectiveCamera):
    """Camera rotating around a target point using spherical coordinates."""

    def __init__(self, target: np.ndarray | None = None, radius: float = 5.0, azimuth: float = 45.0, elevation: float = 30.0, **kwargs):
        pose = kwargs.pop("pose", None)
        super().__init__(pose=pose, **kwargs)
        self.target = np.array(target if target is not None else [0.0, 0.0, 0.0], dtype=np.float32)
        self.radius = radius
        self.azimuth = math.radians(azimuth)
        self.elevation = math.radians(elevation)
        self._min_radius = 1.0
        self._max_radius = 100.0
        self._update_pose()

    def _update_pose(self):
        r = max(self._min_radius, min(self._max_radius, self.radius))
        cos_elev = math.cos(self.elevation)
        eye = np.array(
            [
                self.target[0] + r * math.cos(self.azimuth) * cos_elev,
                self.target[1] + r * math.sin(self.azimuth) * cos_elev,
                self.target[2] + r * math.sin(self.elevation),
            ],
            dtype=np.float32,
        )
        self.pose = Pose3.looking_at(eye=eye, target=self.target)

    def orbit(self, delta_azimuth: float, delta_elevation: float):
        self.azimuth += math.radians(delta_azimuth)
        self.elevation = np.clip(self.elevation + math.radians(delta_elevation), math.radians(-89.0), math.radians(89.0))
        self._update_pose()

    def zoom(self, delta: float):
        self.radius = np.clip(self.radius + delta, self._min_radius, self._max_radius)
        self._update_pose()

    def pan(self, dx: float, dy: float):
        rot = self.pose.rotation_matrix()
        right = rot[:, 0]
        up = rot[:, 1]
        self.target += right * dx + up * dy
        self._update_pose()
