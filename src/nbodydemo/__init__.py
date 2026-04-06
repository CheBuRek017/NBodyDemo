"""NBodyDemo - N-body gravitational simulation with Barnes-Hut octree."""

from .body import Body
from .octree import OctreeNode
from .integrator import compute_accelerations, step_leapfrog, G
from .simulator import Simulator
from .renderer import Renderer
from .bodies_config import get_bodies

__all__ = [
    "Body",
    "OctreeNode",
    "compute_accelerations",
    "step_leapfrog",
    "G",
    "Simulator",
    "Renderer",
    "get_bodies",
]
