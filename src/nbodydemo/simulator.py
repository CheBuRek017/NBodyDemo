"""N-body gravitational simulation engine with adaptive timesteps."""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .integrator import step_leapfrog

if TYPE_CHECKING:
    from .body import Body

logger = logging.getLogger(__name__)


class Simulator:
    """Handles all physics, integration, trails, and simulation time.

    Completely separate from rendering/input (clean MVC style).

    Attributes:
        bodies: List of celestial bodies in the simulation.
        sim_time: Current simulation time in seconds.
        trail_max_age: Maximum age of trail points in seconds (default: 1 year).
    """

    # Constants
    TRAIL_MAX_AGE: float = 3.15576e7  # 1 Earth year in seconds
    G: float = 6.67430e-11  # Gravitational constant

    def __init__(self, bodies: list[Body]) -> None:
        """Initialize the simulator.

        Args:
            bodies: List of celestial bodies to simulate.
        """
        self.bodies: list[Body] = bodies
        self.sim_time: float = 0.0
        self.trail_max_age: float = self.TRAIL_MAX_AGE

        # Initialize time-based trails for every body
        for body in self.bodies:
            body.trail = deque()  # positions
            body.trail_times = deque()  # corresponding sim_time

    def compute_adaptive_dt(self) -> float:
        """Compute adaptive timestep based on Hill-sphere orbital timescales.

        Returns:
            Recommended timestep in seconds.
        """
        num_bodies = len(self.bodies)
        if num_bodies < 2:
            return 3600.0

        positions = np.stack([b.pos for b in self.bodies])
        masses = np.array([b.mass for b in self.bodies], dtype=np.float64)
        rel_pos = positions[None, :, :] - positions[:, None, :]
        dists = np.sqrt(np.sum(rel_pos**2, axis=-1))
        pair_idx1, pair_idx2 = np.triu_indices(num_bodies, k=1)
        pair_dists = dists[pair_idx1, pair_idx2]
        mass_sums = masses[pair_idx1] + masses[pair_idx2]

        orbital_times = np.sqrt(
            pair_dists**3 / (self.G * mass_sums + 1e-30)
        )
        return 0.02 * np.min(orbital_times)

    def step(self, dt: float) -> None:
        """Perform one simulation step.

        Args:
            dt: Time step in seconds.
        """
        if dt <= 0:
            logger.debug("Skipping step with non-positive dt: %s", dt)
            return
        step_leapfrog(self.bodies, dt)
        self.sim_time += dt
        self._update_trails()

    def _update_trails(self) -> None:
        """Append new trail point and cull old points beyond max age."""
        for body in self.bodies:
            if body.trail is None or body.trail_times is None:
                logger.warning("Body %s has uninitialized trails", body.name)
                continue
            body.trail.append(body.pos.copy())
            body.trail_times.append(self.sim_time)
            # Cull old points (keeps trail length stable → FPS stays high)
            while body.trail_times and body.trail_times[0] < self.sim_time - self.trail_max_age:
                body.trail.popleft()
                body.trail_times.popleft()