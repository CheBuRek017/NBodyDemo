"""Leapfrog integrator for symplectic time integration of N-body systems."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .octree import OctreeNode

if TYPE_CHECKING:
    from .body import Body

# Gravitational constant in m^3 kg^-1 s^-2
G: float = 6.67430e-11


def compute_accelerations(bodies: list[Body]) -> list[npt.NDArray[np.float64]]:
    """Compute gravitational accelerations for all bodies using Barnes-Hut octree.

    Args:
        bodies: List of celestial bodies.

    Returns:
        List of acceleration vectors (ax, ay, az) in m/s^2 for each body.
    """
    num_bodies = len(bodies)
    if num_bodies < 2:
        return [np.zeros(3, dtype=np.float64) for _ in bodies]

    positions_array = np.stack([b.pos for b in bodies])
    min_position = np.min(positions_array, axis=0)
    max_position = np.max(positions_array, axis=0)
    tree_center = (min_position + max_position) / 2
    max_extent = np.max(max_position - min_position)
    tree_size = max(max_extent * 1.2, 1e9)

    root = OctreeNode(tree_center, tree_size)
    for body in bodies:
        root.insert(body)

    return [root.compute_force(body, theta=0.5, G=G) for body in bodies]


def step_leapfrog(bodies: list[Body], timestep: float) -> None:
    """Perform one leapfrog integration step.

    The leapfrog method is a symplectic integrator that preserves energy
    better than simple Euler integration for orbital mechanics.

    Args:
        bodies: List of celestial bodies to integrate.
        timestep: Time step in seconds.
    """
    accelerations = compute_accelerations(bodies)
    for body, acceleration in zip(bodies, accelerations):
        body.vel += 0.5 * acceleration * timestep
    for body in bodies:
        body.pos += body.vel * timestep
    accelerations = compute_accelerations(bodies)
    for body, acceleration in zip(bodies, accelerations):
        body.vel += 0.5 * acceleration * timestep