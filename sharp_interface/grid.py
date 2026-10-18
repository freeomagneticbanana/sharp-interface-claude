"""Uniform Cartesian grid. Arrays use 'ij' indexing: axis 0 is x, axis 1 is y."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid2D:
    nx: int
    ny: int
    dx: float
    x0: float = 0.0
    y0: float = 0.0

    @property
    def shape(self):
        return (self.nx, self.ny)

    @property
    def x(self):
        return self.x0 + self.dx * np.arange(self.nx)

    @property
    def y(self):
        return self.y0 + self.dx * np.arange(self.ny)

    def mesh(self):
        return np.meshgrid(self.x, self.y, indexing="ij")

    @classmethod
    def square(cls, n, length=1.0, centered=False):
        dx = length / n
        off = -0.5 * length + 0.5 * dx if centered else 0.5 * dx
        return cls(n, n, dx, off, off)
