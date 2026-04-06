"""Configuration loader for celestial bodies from JSON files."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .body import Body

logger = logging.getLogger(__name__)


def get_bodies(config_path: str | Path | None = None) -> list[Body]:
    """Load all bodies from a JSON configuration file.

    Supports absolute planets and relative moons automatically.

    Args:
        config_path: Path to the JSON configuration file.
            Defaults to project root / bodies.json.

    Returns:
        List of configured Body objects.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If parent body is referenced before definition.
        json.JSONDecodeError: If config file is invalid JSON.
    """
    if config_path is None:
        # points to project root / bodies.json
        config_path = Path(__file__).parent.parent.parent / "bodies.json"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)

    bodies: list[Body] = []
    name_to_body: dict[str, Body] = {}

    for item in data:
        if "relative_to" in item and item["relative_to"]:
            parent_name = item["relative_to"]
            if parent_name not in name_to_body:
                raise ValueError(
                    f"Parent body '{parent_name}' must appear before "
                    f"'{item['name']}' in bodies.json"
                )
            parent = name_to_body[parent_name]
            pos = (
                np.array(parent.pos, dtype=np.float64)
                + np.array(item["relative_position"], dtype=np.float64)
            )
            vel = (
                np.array(parent.vel, dtype=np.float64)
                + np.array(item["relative_velocity"], dtype=np.float64)
            )
        else:
            pos = np.array(item["position"], dtype=np.float64)
            vel = np.array(item["velocity"], dtype=np.float64)

        body = Body(
            name=item["name"],
            mass=item["mass"],
            position=pos,
            velocity=vel,
            radius=item["radius"],
            color=tuple(item["color"]),  # type: ignore[arg-type]
        )
        bodies.append(body)
        name_to_body[item["name"]] = body

    # Center-of-mass correction
    total_mass = sum(b.mass for b in bodies)
    if total_mass == 0:
        logger.warning("Total mass is zero, skipping center-of-mass correction")
        return bodies

    com: npt.NDArray[np.float64] = (
        sum(b.mass * b.pos for b in bodies) / total_mass
    )
    for b in bodies:
        b.pos -= com

    total_momentum = sum(b.mass * b.vel for b in bodies)
    if len(bodies) > 0 and bodies[0].mass > 0:
        bodies[0].vel = -total_momentum / bodies[0].mass

    return bodies