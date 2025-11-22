"""GPU mesh helper built on top of :mod:`termin.mesh` geometry."""

from __future__ import annotations

from typing import Dict

from termin.mesh.mesh import Mesh
from .entity import RenderContext
from .backends.base import MeshHandle


class MeshDrawable:
    """Uploads CPU mesh data to GPU buffers and issues draw commands."""

    def __init__(self, mesh: Mesh):
        self._mesh = mesh
        if self._mesh.vertex_normals is None:
            self._mesh.compute_vertex_normals()
        self._context_resources: Dict[int, MeshHandle] = {}

    def upload(self, context: RenderContext):
        ctx = context.context_key
        if ctx in self._context_resources:
            return
        handle = context.graphics.create_mesh(self._mesh)
        self._context_resources[ctx] = handle

    def draw(self, context: RenderContext):
        ctx = context.context_key
        if ctx not in self._context_resources:
            self.upload(context)
        handle = self._context_resources[ctx]
        handle.draw()

    def delete(self):
        for handle in self._context_resources.values():
            handle.delete()
        self._context_resources.clear()
