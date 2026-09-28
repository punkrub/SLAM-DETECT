"""Look up safe, mapped shooting cells for a requested sign class.

This only plans from recorded map data. It does not move the robot or fire.
"""

import argparse
import json
from pathlib import Path


def find_sign_targets(map_data, color, shape):
    """Return matching signs and their already-visited, verified-open positions."""
    targets = []
    for sign in map_data.get("signs", []):
        if sign.get("color", "").casefold() != color.casefold():
            continue
        if sign.get("shape", "").casefold() != shape.casefold():
            continue
        positions = sign.get("shooting_positions", [])
        if positions:
            targets.append({
                "sign": {
                    "color": sign["color"],
                    "shape": sign["shape"],
                    "cell": sign["cell"],
                    "direction": sign["direction"],
                    "shape_confidence": sign.get("shape_confidence"),
                },
                "shooting_positions": positions,
            })
    return targets


def main():
    parser = argparse.ArgumentParser(
        description="Find mapped 1- or 2-tile shooting positions for a sign"
    )
    parser.add_argument("map", type=Path, help="explored_map.json from a mission")
    parser.add_argument("--color", required=True,
                        choices=("Red", "Green", "Blue", "Yellow"))
    parser.add_argument("--shape", required=True,
                        choices=("Circle", "Square", "Horizontal_Rect", "Vertical_Rect"))
    args = parser.parse_args()

    data = json.loads(args.map.read_text(encoding="utf-8"))
    matches = find_sign_targets(data, args.color, args.shape)
    print(json.dumps({"target": {"color": args.color, "shape": args.shape},
                      "matches": matches}, indent=2))
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
