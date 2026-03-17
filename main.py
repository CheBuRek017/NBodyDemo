"""
NBodyDemo – Modern modular version
Run with: python main.py
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path so we can import without installing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from nbodydemo.body import Body
from nbodydemo.renderer import Renderer

# ==============================================================
# Solar System + major moons (realistic initial conditions)
# ==============================================================
bodies = [
    Body("Sun",     1.98847e30, [0.0, 0.0, 0.0],          [0.0, 0.0, 0.0],          6.9634e8, (1.0, 0.85, 0.0)),
    Body("Mercury", 3.3011e23,  [5.79e10, 0.0, 0.0],      [0.0, 4.79e4, 0.0],       2.44e6,   (0.7, 0.7, 0.7)),
    Body("Venus",   4.8675e24,  [1.082e11, 0.0, 0.0],     [0.0, 3.5e4, 0.0],        6.05e6,   (1.0, 0.9, 0.6)),
    Body("Earth",   5.97237e24,[1.496e11, 0.0, 0.0],     [0.0, 2.978e4, 0.0],      6.371e6,  (0.0, 0.5, 1.0)),
    Body("Moon",    7.342e22,   [1.496e11 + 3.844e8, 0.0, 0.0], [0.0, 2.978e4 + 1.022e3, 0.0], 1.737e6,  (0.8, 0.8, 0.8)),
    Body("Mars",    6.4171e23,  [2.279e11, 0.0, 0.0],    [0.0, 2.41e4, 0.0],       3.389e6,  (1.0, 0.4, 0.3)),
    Body("Jupiter", 1.8982e27,  [7.785e11, 0.0, 0.0],    [0.0, 1.307e4, 0.0],      6.9911e7, (1.0, 0.75, 0.5)),
    Body("Saturn",  5.6834e26,  [1.4335e12, 0.0, 0.0],   [0.0, 9.68e3, 0.0],       5.8232e7, (0.95, 0.9, 0.7)),
    Body("Uranus",  8.6810e25,  [2.8725e12, 0.0, 0.0],   [0.0, 6.8e3, 0.0],        2.5362e7, (0.6, 0.8, 1.0)),
    Body("Neptune", 1.02413e26, [4.4951e12, 0.0, 0.0],   [0.0, 5.4e3, 0.0],        2.4622e7, (0.4, 0.6, 1.0)),
]

# Jupiter moons
jupiter = bodies[6]
bodies.extend([
    Body("Io",       8.9319e22, jupiter.pos + np.array([4.2170e8, 0.0, 0.0]), jupiter.vel + np.array([0.0, 1.7334e4, 0.0]), 1.8216e6, (1.0, 0.8, 0.6)),
    Body("Europa",   4.7998e22, jupiter.pos + np.array([6.7090e8, 0.0, 0.0]), jupiter.vel + np.array([0.0, 1.3740e4, 0.0]), 1.5608e6, (0.8, 0.9, 1.0)),
    Body("Ganymede", 1.4819e23, jupiter.pos + np.array([1.0704e9, 0.0, 0.0]), jupiter.vel + np.array([0.0, 1.0880e4, 0.0]), 2.631e6,   (0.7, 0.7, 0.7)),
    Body("Callisto", 1.0759e23, jupiter.pos + np.array([1.8827e9, 0.0, 0.0]), jupiter.vel + np.array([0.0, 8.204e3,  0.0]), 2.410e6,   (0.6, 0.6, 0.6)),
])

# Saturn moon
saturn = bodies[7]
bodies.append(Body("Titan", 1.3452e23, saturn.pos + np.array([1.2219e9, 0.0, 0.0]), saturn.vel + np.array([0.0, 5.57e3, 0.0]), 2.575e6, (0.8, 0.8, 0.7)))

# Mars moons
mars_body = bodies[5]
bodies.extend([
    Body("Phobos", 1.0659e16, mars_body.pos + np.array([9.377e6, 0.0, 0.0]), mars_body.vel + np.array([0.0, 2.138e3, 0.0]), 1.13e4, (0.5, 0.5, 0.5)),
    Body("Deimos", 1.4762e15, mars_body.pos + np.array([2.346e7, 0.0, 0.0]), mars_body.vel + np.array([0.0, 1.351e3, 0.0]), 6.2e3,  (0.6, 0.6, 0.6)),
])

bodies.append(Body("Ceres", 9.3835e20, [4.14e11, 0.0, 0.0], [0.0, 1.79e4, 0.0], 4.73e5, (0.6, 0.4, 0.3)))

# Center-of-mass correction (so Sun doesn't fly off)
total_mass = sum(body.mass for body in bodies)
center_of_mass = sum(body.mass * body.pos for body in bodies) / total_mass
for body in bodies:
    body.pos -= center_of_mass

total_momentum = sum(body.mass * body.vel for body in bodies)
bodies[0].vel = -total_momentum / bodies[0].mass

if __name__ == "__main__":
    renderer = Renderer(bodies)
    renderer.run()