"""Extract named scene-object world transforms from a Unity YAML scene."""

from __future__ import annotations

"""Inspect selected Unity scene objects and export their world coordinates."""

import argparse
import csv
import re
from pathlib import Path

import numpy as np


DOC_RE = re.compile(r"^--- !u!(\d+) &(-?\d+)(?: stripped)?\s*$", re.MULTILINE)


def field(text: str, key: str, default: str = "") -> str:
    match = re.search(rf"^  {re.escape(key)}:\s*(.*?)\s*$", text, re.MULTILINE)
    return match.group(1) if match else default


def file_id(text: str, key: str) -> int:
    match = re.search(rf"^  {re.escape(key)}:\s*\{{fileID:\s*(-?\d+)", text, re.MULTILINE)
    return int(match.group(1)) if match else 0


def vec(text: str, key: str, n: int, default: tuple[float, ...]) -> np.ndarray:
    match = re.search(rf"^  {re.escape(key)}:\s*\{{([^}}]+)\}}", text, re.MULTILINE)
    if not match:
        return np.asarray(default, dtype=float)
    pairs = dict(re.findall(r"([xyzw]):\s*([-+0-9.eE]+)", match.group(1)))
    labels = "xyzw"[:n]
    return np.asarray([float(pairs.get(k, default[i])) for i, k in enumerate(labels)], dtype=float)


def quat_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    norm = np.linalg.norm(q)
    if norm == 0:
        return np.eye(3)
    x, y, z, w = q / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def local_matrix(position: np.ndarray, rotation: np.ndarray, scale: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = quat_matrix(rotation) @ np.diag(scale)
    matrix[:3, 3] = position
    return matrix


def parse_scene(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    matches = list(DOC_RE.finditer(text))
    game_objects: dict[int, str] = {}
    transforms: dict[int, dict[str, object]] = {}

    for index, match in enumerate(matches):
        class_id = int(match.group(1))
        object_id = int(match.group(2))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.end():end]
        if class_id == 1:
            name = field(block, "m_Name")
            game_objects[object_id] = name.strip('"')
        elif class_id in {4, 224}:
            transforms[object_id] = {
                "game_object": file_id(block, "m_GameObject"),
                "parent": file_id(block, "m_Father"),
                "position": vec(block, "m_LocalPosition", 3, (0.0, 0.0, 0.0)),
                "rotation": vec(block, "m_LocalRotation", 4, (0.0, 0.0, 0.0, 1.0)),
                "scale": vec(block, "m_LocalScale", 3, (1.0, 1.0, 1.0)),
            }

    cache: dict[int, np.ndarray] = {}

    def world_matrix(transform_id: int, stack: set[int] | None = None) -> np.ndarray:
        if transform_id in cache:
            return cache[transform_id]
        if transform_id not in transforms:
            return np.eye(4)
        stack = set() if stack is None else stack
        if transform_id in stack:
            return np.eye(4)
        stack.add(transform_id)
        item = transforms[transform_id]
        local = local_matrix(item["position"], item["rotation"], item["scale"])
        parent = int(item["parent"])
        world = world_matrix(parent, stack) @ local if parent else local
        cache[transform_id] = world
        return world

    rows: list[dict[str, object]] = []
    for transform_id, item in transforms.items():
        game_object_id = int(item["game_object"])
        name = game_objects.get(game_object_id, "")
        if not name:
            continue
        world = world_matrix(transform_id)
        parent_id = int(item["parent"])
        parent_name = ""
        if parent_id in transforms:
            parent_name = game_objects.get(int(transforms[parent_id]["game_object"]), "")
        rows.append({
            "game_object_id": game_object_id,
            "transform_id": transform_id,
            "name": name,
            "parent_name": parent_name,
            "world_x": world[0, 3],
            "world_y": world[1, 3],
            "world_z": world[2, 3],
            "local_x": float(item["position"][0]),
            "local_y": float(item["position"][1]),
            "local_z": float(item["position"][2]),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--pattern",
        default=r"gate|stair|exit|hallfloor|platform|turnstile|ticket|shelter",
    )
    args = parser.parse_args()
    rows = parse_scene(args.scene)
    pattern = re.compile(args.pattern, re.IGNORECASE)
    selected = [row for row in rows if pattern.search(str(row["name"]))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]) if selected else ["name"])
        writer.writeheader()
        writer.writerows(selected)
    print(f"scene objects={len(rows)}, selected={len(selected)}, output={args.output}")


if __name__ == "__main__":
    main()
