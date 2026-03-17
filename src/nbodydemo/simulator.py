import numpy as np
from collections import deque
from .integrator import step_leapfrog
from .body import Body

class Simulator:
    """Handles all physics, integration, trails, and simulation time.
    Completely separate from rendering/input (clean MVC style)."""

    def __init__(self, bodies: list[Body]):
        self.bodies = bodies
        self.sim_time = 0.0
        self.trail_max_age = 3.15576e7  # 1 Earth year in seconds → dynamic length!

        # Initialize time-based trails for every body
        for body in self.bodies:
            body.trail = deque()          # positions
            body.trail_times = deque()    # corresponding sim_time

    def compute_adaptive_dt(self) -> float:
        """Hill-sphere adaptive timestep (moved from Renderer)."""
        num_bodies = len(self.bodies)
        if num_bodies < 2:
            return 3600.0

        positions = np.stack([b.pos for b in self.bodies])
        masses = np.array([b.mass for b in self.bodies], dtype=np.float64)
        rel_pos = positions[None, :, :] - positions[:, None, :]
        dists = np.sqrt(np.sum(rel_pos ** 2, axis=-1))
        pair_idx1, pair_idx2 = np.triu_indices(num_bodies, k=1)
        pair_dists = dists[pair_idx1, pair_idx2]
        mass_sums = masses[pair_idx1] + masses[pair_idx2]

        orbital_times = np.sqrt(pair_dists ** 3 / (6.67430e-11 * mass_sums + 1e-30))
        return 0.02 * np.min(orbital_times)

    def step(self, dt: float):
        """One physics step + trail update."""
        if dt <= 0:
            return
        step_leapfrog(self.bodies, dt)
        self.sim_time += dt
        self._update_trails()

    def _update_trails(self):
        """Append new point + automatically cull anything older than 1 year."""
        for body in self.bodies:
            body.trail.append(body.pos.copy())
            body.trail_times.append(self.sim_time)
            # Cull old points (keeps trail length stable → FPS stays high)
            while body.trail_times and body.trail_times[0] < self.sim_time - self.trail_max_age:
                body.trail.popleft()
                body.trail_times.popleft()