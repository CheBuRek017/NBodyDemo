"""Body module for celestial body representation."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence


class Body:
    """Single celestial body with position, velocity, and visual data.

    Attributes:
        name: Name of the celestial body.
        mass: Mass in kilograms.
        pos: Position vector (x, y, z) in meters.
        vel: Velocity vector (vx, vy, vz) in m/s.
        radius: Physical radius in meters.
        color: RGB color tuple (r, g, b) with values in [0, 1].
        trail: Deque of historical positions for rendering trails.
        trail_times: Deque of simulation times corresponding to trail positions.
    """

    def __init__(
        self,
        name: str,
        mass: float,
        position: Sequence[float],
        velocity: Sequence[float],
        radius: float,
        color: tuple[float, float, float],
    ) -> None:
        """Initialize a celestial body.

        Args:
            name: Name of the celestial body.
            mass: Mass in kilograms.
            position: Initial position (x, y, z) in meters.
            velocity: Initial velocity (vx, vy, vz) in m/s.
            radius: Physical radius in meters.
            color: RGB color tuple (r, g, b) with values in [0, 1].
        """
        self.name: str = name
        self.mass: float = mass
        self.pos: npt.NDArray[np.float64] = np.array(position, dtype=np.float64)
        self.vel: npt.NDArray[np.float64] = np.array(velocity, dtype=np.float64)
        self.radius: float = radius
        self.color: tuple[float, float, float] = color
        self.trail: deque[npt.NDArray[np.float64]] | None = None
        self.trail_times: deque[float] | None = None