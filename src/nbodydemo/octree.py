"""Barnes-Hut octree implementation for O(n log n) force approximation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .body import Body


class OctreeNode:
    """Octree node for Barnes-Hut gravitational force computation.

    Attributes:
        center: Center position of this octant (x, y, z) in meters.
        size: Side length of this cubic octant in meters.
        mass: Total mass contained in this node in kilograms.
        com: Center of mass position (x, y, z) in meters.
        children: List of 8 child nodes (None if leaf).
        is_leaf: True if this node has no children.
        body: Body contained in this node (only for leaves with single body).
    """

    def __init__(self, center: list[float] | npt.NDArray[np.float64], size: float) -> None:
        """Initialize an octree node.

        Args:
            center: Center position (x, y, z) in meters.
            size: Side length of the cubic region in meters.
        """
        self.center: npt.NDArray[np.float64] = np.array(center, dtype=np.float64)
        self.size: float = size
        self.mass: float = 0.0
        self.com: npt.NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        self.children: list[OctreeNode | None] = [None] * 8
        self.is_leaf: bool = True
        self.body: Body | None = None

    def _get_octant(self, position: npt.NDArray[np.float64]) -> int:
        """Determine which octant a position falls into.

        Args:
            position: Position vector to classify.

        Returns:
            Octant index (0-7).
        """
        octant_index = 0
        if position[0] >= self.center[0]:
            octant_index |= 1
        if position[1] >= self.center[1]:
            octant_index |= 2
        if position[2] >= self.center[2]:
            octant_index |= 4
        return octant_index

    def _subdivide(self) -> None:
        """Subdivide this node into 8 child octants."""
        quarter = self.size / 4.0
        half_size = self.size / 2.0
        for octant_index in range(8):
            x_offset = quarter if (octant_index & 1) else -quarter
            y_offset = quarter if (octant_index & 2) else -quarter
            z_offset = quarter if (octant_index & 4) else -quarter
            child_center = self.center + np.array(
                [x_offset, y_offset, z_offset], dtype=np.float64
            )
            self.children[octant_index] = OctreeNode(child_center, half_size)

    def _insert_to_child(self, body: Body) -> None:
        """Insert a body into the appropriate child octant.

        Args:
            body: The celestial body to insert.
        """
        octant_index = self._get_octant(body.pos)
        child = self.children[octant_index]
        if child is None:
            raise RuntimeError(f"Child node {octant_index} should exist but is None")
        child.insert(body)

    def insert(self, body: Body) -> None:
        """Insert a body into this octree node.

        Args:
            body: The celestial body to insert.
        """
        if self.mass == 0.0:
            self.body = body
            self.mass = body.mass
            self.com = body.pos.copy()
            return

        if self.is_leaf and self.body is not None:
            self._subdivide()
            old_body = self.body
            self.body = None
            self.is_leaf = False
            self._insert_to_child(old_body)

        if not self.is_leaf:
            self._insert_to_child(body)

        old_mass = self.mass
        self.mass += body.mass
        if self.mass > 0:
            self.com = (self.com * old_mass + body.pos * body.mass) / self.mass

    def compute_force(
        self, body: Body, theta: float, G: float
    ) -> npt.NDArray[np.float64]:
        """Compute gravitational force on a body using Barnes-Hut approximation.

        Args:
            body: The body to compute force on.
            theta: Opening angle threshold for approximation (typically 0.5).
            G: Gravitational constant.

        Returns:
            Force vector (Fx, Fy, Fz) in Newtons.
        """
        if self.mass == 0.0 or (self.is_leaf and self.body is body):
            return np.zeros(3, dtype=np.float64)

        com_to_body_vector = self.com - body.pos
        distance_squared = np.dot(com_to_body_vector, com_to_body_vector)
        if distance_squared < 1e6:
            return np.zeros(3, dtype=np.float64)
        distance = np.sqrt(distance_squared)

        if self.is_leaf or (self.size / distance < theta):
            inverse_distance_cubed = 1.0 / (distance_squared * distance)
            return G * self.mass * com_to_body_vector * inverse_distance_cubed
        else:
            total_acceleration = np.zeros(3, dtype=np.float64)
            for child in self.children:
                if child is not None:
                    total_acceleration += child.compute_force(body, theta, G)
            return total_acceleration