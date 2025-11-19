"""Simple 2D texture wrapper for GLFW/PyOpenGL stack."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from OpenGL import GL as gl


class Texture:
    """Loads an image via Pillow and uploads it as ``GL_TEXTURE_2D``."""

    def __init__(self, path: Optional[str | Path] = None):
        self.handle = None
        self._image_data: Optional[np.ndarray] = None
        self._size: Optional[tuple[int, int]] = None
        if path is not None:
            self.load(path)

    def load(self, path: str | Path):
        image = Image.open(path).convert("RGBA")
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        data = np.array(image, dtype=np.uint8)
        width, height = image.size

        self._image_data = data
        self._size = (width, height)
        self.handle = None  # Mark for upload in the next bind.

    def _upload_if_needed(self):
        if self.handle is not None:
            return
        if self._image_data is None or self._size is None:
            raise RuntimeError("Texture has no image data to upload.")
        self.handle = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.handle)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA,
            self._size[0],
            self._size[1],
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            self._image_data,
        )
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)

    def bind(self, unit: int = 0):
        self._upload_if_needed()
        gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.handle or 0)

    @classmethod
    def from_file(cls, path: str | Path) -> "Texture":
        tex = cls()
        tex.load(path)
        return tex
