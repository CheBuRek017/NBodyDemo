import numpy as np

class Body:
    """Single celestial body with position, velocity, and visual data."""
    def __init__(self, name, mass, position, velocity, radius, color):
        self.name = name
        self.mass = mass
        self.pos = np.array(position, dtype=np.float64)
        self.vel = np.array(velocity, dtype=np.float64)
        self.radius = radius
        self.color = color
        self.trail = None
        self.trail_times = None