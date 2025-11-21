# termin/visualization/ui/font.py
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from OpenGL import GL as gl

import os

class FontTextureAtlas:
    def __init__(self, path: str, size: int = 32):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Font file not found: {path}")

        self.font = ImageFont.truetype(path, size)
        self.size = size
        self.glyphs = {}
        self.texture = None
        self.tex_w = 0
        self.tex_h = 0
        self._build_atlas()

    def _build_atlas(self):
        chars = [chr(i) for i in range(32, 127)]
        padding = 2

        glyph_images = []
        max_w = max_h = 0
        for ch in chars:
            w, h = self.font.getsize(ch)
            img = Image.new("L", (w, h))
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), ch, fill=255, font=self.font)
            glyph_images.append((ch, img))
            max_w = max(max_w, w)
            max_h = max(max_h, h)

        cols = 16
        rows = (len(chars) + cols - 1) // cols
        atlas_w = cols * (max_w + padding)
        atlas_h = rows * (max_h + padding)
        self.tex_w = atlas_w
        self.tex_h = atlas_h

        atlas = Image.new("L", (atlas_w, atlas_h))
        draw = ImageDraw.Draw(atlas)

        x = y = 0
        for i, (ch, img) in enumerate(glyph_images):
            atlas.paste(img, (x, y))
            w, h = img.size
            self.glyphs[ch] = {
                "uv": (
                    x / atlas_w,
                    y / atlas_h,
                    (x + w) / atlas_w,
                    (y + h) / atlas_h
                ),
                "size": (w, h)
            }
            x += max_w + padding
            if (i + 1) % cols == 0:
                x = 0
                y += max_h + padding

        data = np.array(atlas, dtype=np.uint8)
        self.texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RED,
            atlas_w, atlas_h, 0,
            gl.GL_RED, gl.GL_UNSIGNED_BYTE, data
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
