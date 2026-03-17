import json
from pathlib import Path
import numpy as np
from .body import Body

def get_bodies(config_path: str | Path | None = None) -> list[Body]:
    """Load all bodies from bodies.json (data-driven config).
    Supports absolute planets + relative moons automatically."""
    if config_path is None:
        # points to project root / bodies.json
        config_path = Path(__file__).parent.parent.parent / "bodies.json"

    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)

    bodies: list[Body] = []
    name_to_body: dict[str, Body] = {}

    for item in data:
        if "relative_to" in item and item["relative_to"]:
            parent_name = item["relative_to"]
            if parent_name not in name_to_body:
                raise ValueError(f"Parent body '{parent_name}' must appear before '{item['name']}' in bodies.json")
            parent = name_to_body[parent_name]
            pos = np.array(parent.pos, dtype=np.float64) + np.array(item["relative_position"], dtype=np.float64)
            vel = np.array(parent.vel, dtype=np.float64) + np.array(item["relative_velocity"], dtype=np.float64)
        else:
            pos = np.array(item["position"], dtype=np.float64)
            vel = np.array(item["velocity"], dtype=np.float64)

        body = Body(
            name=item["name"],
            mass=item["mass"],
            position=pos,
            velocity=vel,
            radius=item["radius"],
            color=tuple(item["color"])
        )
        bodies.append(body)
        name_to_body[item["name"]] = body

    # Center-of-mass correction
    total_mass = sum(b.mass for b in bodies)
    com = sum(b.mass * b.pos for b in bodies) / total_mass
    for b in bodies:
        b.pos -= com

    total_momentum = sum(b.mass * b.vel for b in bodies)
    bodies[0].vel = -total_momentum / bodies[0].mass

    return bodies