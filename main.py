import sys
from pathlib import Path

# Fallback for people who didn't run `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent / "src"))

from nbodydemo.bodies_config import get_bodies
from nbodydemo.renderer import Renderer

if __name__ == "__main__":
    bodies = get_bodies()
    renderer = Renderer(bodies)
    renderer.run()