import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from nbodydemo.bodies_config import get_bodies
from nbodydemo.simulator import Simulator
from nbodydemo.renderer import Renderer

if __name__ == "__main__":
    bodies = get_bodies()
    simulator = Simulator(bodies)
    renderer = Renderer(simulator)
    renderer.run()