#!/usr/bin/env python3
"""2.5D semantic-zone query interface for Unity world coordinates.

The query follows the formal Unity partition:

1. Use Unity Y to select Platform or Hall.
2. Test Unity (X, Z) against the selected floor's collider rectangles.
3. Resolve shared boundaries with the formal zone priority.

The public ``classify_point(x, y, z)`` method returns a zone-id string and is
intentionally similar to the interface in ``plot_experiment_zones.py``.
Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DEFAULT_GEOMETRY_PATH = (
    Path(__file__).resolve().parent
    / "Results_Guidance_Spatial_Causal_Analysis"
    / "zone_annotation"
    / "semantic_zones.json"
)


@dataclass(frozen=True)
class ColliderPart:
    """One axis-aligned X-Z rectangle used by a Unity BoxCollider."""

    xmin: float
    xmax: float
    zmin: float
    zmax: float

    def contains_xz(self, x: float, z: float, epsilon: float = 0.0) -> bool:
        return (
            self.xmin - epsilon <= x <= self.xmax + epsilon
            and self.zmin - epsilon <= z <= self.zmax + epsilon
        )


@dataclass(frozen=True)
class SemanticZone:
    """Immutable query representation of one formal semantic zone."""

    zone_id: str
    zone_type: str
    floor_id: str
    priority: float
    y_min: float
    y_max: float
    area_xz: float
    collider_parts: tuple[ColliderPart, ...]

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "SemanticZone":
        center_y = float(record["center_y"])
        size_y = float(record["size_y"])
        raw_parts = record.get("collider_parts")
        if not isinstance(raw_parts, list) or not raw_parts:
            raise ValueError(f"Zone {record.get('zone_id')!r} has no collider_parts")

        parts = tuple(
            ColliderPart(
                xmin=float(part["xmin"]),
                xmax=float(part["xmax"]),
                zmin=float(part["zmin"]),
                zmax=float(part["zmax"]),
            )
            for part in raw_parts
        )
        return cls(
            zone_id=str(record["zone_id"]),
            zone_type=str(record["zone_type"]),
            floor_id=str(record["floor_id"]),
            priority=float(record.get("priority", 0.0)),
            y_min=float(record.get("y_min", center_y - size_y / 2.0)),
            y_max=float(record.get("y_max", center_y + size_y / 2.0)),
            area_xz=float(
                record.get(
                    "area_xz",
                    sum(
                        (part.xmax - part.xmin) * (part.zmax - part.zmin)
                        for part in parts
                    ),
                )
            ),
            collider_parts=parts,
        )

    def belongs_to_floor(self, floor_name: str) -> bool:
        """Return whether this zone participates in a 2.5D floor query."""
        if self.floor_id == floor_name:
            return True
        # Cross-floor Stair records use floor_id == "Hall-Platform".
        return floor_name in self.floor_id.split("-")

    def contains(self, x: float, y: float, z: float, epsilon: float) -> bool:
        if not (self.y_min - epsilon <= y <= self.y_max + epsilon):
            return False
        return any(part.contains_xz(x, z, epsilon) for part in self.collider_parts)


class SemanticZoneQuery:
    """Query the formal semantic partition with Unity ``(x, y, z)`` points.

    Parameters
    ----------
    geometry_path:
        Path to ``semantic_zones.json``.  The current formal file is
        used by default.
    split_y:
        Platform/Hall decision height.  If omitted, it is derived from the
        shared boundary in the geometry file (currently ``-11.8``).
    null_name:
        String returned when the point is outside every zone.
    epsilon:
        Inclusive boundary tolerance in Unity metres.
    """

    PLATFORM = "Platform"
    HALL = "Hall"

    def __init__(
        self,
        geometry_path: str | Path = DEFAULT_GEOMETRY_PATH,
        *,
        split_y: float | None = None,
        null_name: str = "Null",
        epsilon: float = 1e-7,
    ) -> None:
        self.geometry_path = Path(geometry_path).expanduser().resolve()
        self.null_name = str(null_name)
        self.epsilon = float(epsilon)
        self.zones = self._load_zones(self.geometry_path)

        platform_zones = [z for z in self.zones if z.floor_id == self.PLATFORM]
        hall_zones = [z for z in self.zones if z.floor_id == self.HALL]
        if not platform_zones or not hall_zones:
            raise ValueError("Geometry must contain both Platform and Hall zones")

        self.platform_floor_y = min(z.y_min for z in platform_zones)
        self.hall_ceiling_y = max(z.y_max for z in hall_zones)
        derived_split = max(z.y_max for z in platform_zones)
        self.split_y = float(derived_split if split_y is None else split_y)

        # Higher-priority zones own shared boundaries.  Smaller area and zone_id
        # are stable tie-breakers for zones with equal formal priority.
        ordered = tuple(
            sorted(self.zones, key=lambda z: (-z.priority, z.area_xz, z.zone_id))
        )
        self._zones_by_floor = {
            self.PLATFORM: tuple(z for z in ordered if z.belongs_to_floor(self.PLATFORM)),
            self.HALL: tuple(z for z in ordered if z.belongs_to_floor(self.HALL)),
        }

    @staticmethod
    def _load_zones(path: Path) -> tuple[SemanticZone, ...]:
        if not path.is_file():
            raise FileNotFoundError(f"Semantic-zone geometry not found: {path}")
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("Expected semantic_zones.json to contain a list")
        zones = tuple(SemanticZone.from_record(record) for record in records)
        if not zones:
            raise ValueError("Semantic-zone geometry is empty")
        zone_ids = [zone.zone_id for zone in zones]
        if len(zone_ids) != len(set(zone_ids)):
            raise ValueError("Semantic-zone geometry contains duplicate zone_id values")
        return zones

    def classify_floor(self, y: float) -> str | None:
        """Classify Unity Y as ``Platform``, ``Hall``, or outside the model."""
        y = float(y)
        if y < self.platform_floor_y - self.epsilon:
            return None
        if y > self.hall_ceiling_y + self.epsilon:
            return None
        return self.PLATFORM if y < self.split_y else self.HALL

    def find_matches(self, x: float, y: float, z: float) -> tuple[SemanticZone, ...]:
        """Return all matches in priority order, mainly for diagnostics."""
        x, y, z = float(x), float(y), float(z)
        floor_name = self.classify_floor(y)
        if floor_name is None:
            return ()
        return tuple(
            zone
            for zone in self._zones_by_floor[floor_name]
            if zone.contains(x, y, z, self.epsilon)
        )

    def classify_point(self, x: float, y: float, z: float) -> str:
        """Return the semantic ``zone_id`` for a Unity ``(x, y, z)`` point."""
        matches = self.find_matches(x, y, z)
        return matches[0].zone_id if matches else self.null_name

    # Explicit alias for call sites that prefer the Unity-specific name.
    classify_unity_point = classify_point

    def classify_position(self, position: Sequence[float]) -> str:
        """Classify a three-item Unity position sequence."""
        if len(position) != 3:
            raise ValueError("Unity position must contain exactly (x, y, z)")
        return self.classify_point(position[0], position[1], position[2])

    def classify_many(self, positions: Iterable[Sequence[float]]) -> list[str]:
        """Classify multiple Unity positions while preserving input order."""
        return [self.classify_position(position) for position in positions]

    def __call__(self, x: float, y: float, z: float) -> str:
        return self.classify_point(x, y, z)


# Module-level convenience interface, matching the supplied example script.
_DEFAULT_QUERY: SemanticZoneQuery | None = None


def get_default_query() -> SemanticZoneQuery:
    global _DEFAULT_QUERY
    if _DEFAULT_QUERY is None:
        _DEFAULT_QUERY = SemanticZoneQuery()
    return _DEFAULT_QUERY


def classify_point(x: float, y: float, z: float) -> str:
    """Classify one Unity point using the current formal partition."""
    return get_default_query().classify_point(x, y, z)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("x", type=float, help="Unity X coordinate")
    parser.add_argument("y", type=float, help="Unity Y coordinate")
    parser.add_argument("z", type=float, help="Unity Z coordinate")
    parser.add_argument(
        "--geometry",
        type=Path,
        default=DEFAULT_GEOMETRY_PATH,
        help="Path to semantic_zones.json",
    )
    args = parser.parse_args()
    query = SemanticZoneQuery(args.geometry)
    print(query.classify_point(args.x, args.y, args.z))


if __name__ == "__main__":
    main()
