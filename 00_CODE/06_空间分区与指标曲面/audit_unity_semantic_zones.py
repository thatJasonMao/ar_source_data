#!/usr/bin/env python3
"""从序列化的 Unity 场景中提取并审计语义区域几何。

用途说明：这是“Unity 场景 → 区域源数据”的只读提取工具。它优先读取
ProBuilder PolyShape 子物体的顶点与挤出高度；没有 PolyShape 时读取父物体
的 BoxCollider，并输出世界坐标、重合、正交性和连通性审计结果。

The script is deliberately read-only with respect to Unity.  It treats a
ProBuilder PolyShape child as the authoritative footprint of its parent zone;
otherwise it uses the parent's BoxCollider.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict, deque
from pathlib import Path


HEADER_RE = re.compile(r"^--- !u!(\d+) &(-?\d+)(?: stripped)?$", re.MULTILINE)
REF_RE = re.compile(r"\{fileID: (-?\d+)")
VEC3_RE = re.compile(
    r"\{x: ([^,}]+), y: ([^,}]+), z: ([^,}]+)(?:, w: ([^,}]+))?\}"
)


def parse_documents(text: str) -> dict[int, tuple[int, str]]:
    matches = list(HEADER_RE.finditer(text))
    docs: dict[int, tuple[int, str]] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        docs[int(match.group(2))] = (int(match.group(1)), text[match.end() : end])
    return docs


def field_text(body: str, name: str, default: str = "") -> str:
    match = re.search(rf"^  {re.escape(name)}: ?(.*)$", body, re.MULTILINE)
    return match.group(1).strip() if match else default


def field_ref(body: str, name: str) -> int:
    raw = field_text(body, name)
    match = REF_RE.search(raw)
    return int(match.group(1)) if match else 0


def parse_vec(raw: str, dims: int = 3) -> tuple[float, ...]:
    match = VEC3_RE.search(raw)
    if not match:
        return (0.0,) * dims
    values = [float(match.group(i)) for i in range(1, dims + 1)]
    return tuple(values)


def parse_block_refs(body: str, key: str) -> list[int]:
    match = re.search(
        rf"^  {re.escape(key)}:\s*\n((?:  - .*\n?)*)", body, re.MULTILINE
    )
    if not match:
        return []
    return [int(value) for value in re.findall(r"fileID: (-?\d+)", match.group(1))]


def parse_points(body: str) -> list[tuple[float, float, float]]:
    match = re.search(
        r"^  m_Points:[ \t]*\r?\n((?:  - \{x: .*\}[ \t]*\r?\n?)*)", body, re.MULTILINE
    )
    if not match:
        return []
    return [parse_vec(raw) for raw in re.findall(r"\{x: .*?\}", match.group(1))]


def identity() -> list[list[float]]:
    return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
            for i in range(4)]


def trs(position: tuple[float, float, float], quaternion: tuple[float, float, float, float],
        scale: tuple[float, float, float]) -> list[list[float]]:
    x, y, z, w = quaternion
    norm = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    rotation = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    matrix = identity()
    for i in range(3):
        for j in range(3):
            matrix[i][j] = rotation[i][j] * scale[j]
        matrix[i][3] = position[i]
    return matrix


def transform_point(matrix: list[list[float]], point: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = point
    values = [x, y, z, 1.0]
    return tuple(sum(matrix[i][j] * values[j] for j in range(4)) for i in range(3))


def polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(sum(points[i][0] * points[(i + 1) % len(points)][1]
                   - points[(i + 1) % len(points)][0] * points[i][1]
                   for i in range(len(points))) / 2.0)


def polygon_centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    signed_twice_area = sum(
        points[i][0] * points[(i + 1) % len(points)][1]
        - points[(i + 1) % len(points)][0] * points[i][1]
        for i in range(len(points))
    )
    if abs(signed_twice_area) < 1e-12:
        return (sum(p[0] for p in points) / len(points),
                sum(p[1] for p in points) / len(points))
    cx = cy = 0.0
    for i, p1 in enumerate(points):
        p2 = points[(i + 1) % len(points)]
        cross = p1[0] * p2[1] - p2[0] * p1[1]
        cx += (p1[0] + p2[0]) * cross
        cy += (p1[1] + p2[1]) * cross
    return cx / (3 * signed_twice_area), cy / (3 * signed_twice_area)


def signed_area(points: list[tuple[float, float]]) -> float:
    return sum(points[i][0] * points[(i + 1) % len(points)][1]
               - points[(i + 1) % len(points)][0] * points[i][1]
               for i in range(len(points))) / 2.0


def clip_polygon(subject: list[tuple[float, float]], rect: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    """Clip a polygon to an axis-aligned rectangle (Sutherland-Hodgman)."""
    xmin, xmax, zmin, zmax = rect
    boundaries = [
        (lambda p: p[0] >= xmin, lambda a, b: (xmin, a[1] + (b[1] - a[1]) * (xmin - a[0]) / (b[0] - a[0]))),
        (lambda p: p[0] <= xmax, lambda a, b: (xmax, a[1] + (b[1] - a[1]) * (xmax - a[0]) / (b[0] - a[0]))),
        (lambda p: p[1] >= zmin, lambda a, b: (a[0] + (b[0] - a[0]) * (zmin - a[1]) / (b[1] - a[1]), zmin)),
        (lambda p: p[1] <= zmax, lambda a, b: (a[0] + (b[0] - a[0]) * (zmax - a[1]) / (b[1] - a[1]), zmax)),
    ]
    output = subject[:]
    for inside, intersection in boundaries:
        if not output:
            break
        input_points, output = output, []
        previous = input_points[-1]
        previous_inside = inside(previous)
        for current in input_points:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    output.append(intersection(previous, current))
                output.append(current)
            elif previous_inside:
                output.append(intersection(previous, current))
            previous, previous_inside = current, current_inside
    return output


def is_axis_aligned_rectangle(points: list[tuple[float, float]], tolerance: float = 1e-4) -> bool:
    if len(points) != 4:
        return False
    return all(min(abs(points[(i + 1) % 4][0] - points[i][0]),
                   abs(points[(i + 1) % 4][1] - points[i][1])) <= tolerance
               for i in range(4))


def bounds(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    zs = [p[1] for p in points]
    return min(xs), max(xs), min(zs), max(zs)


def overlap_area(a: dict, b: dict) -> float:
    ax0, ax1, az0, az1 = a["bounds"]
    bx0, bx1, bz0, bz1 = b["bounds"]
    ix0, ix1 = max(ax0, bx0), min(ax1, bx1)
    iz0, iz1 = max(az0, bz0), min(az1, bz1)
    if ix1 <= ix0 or iz1 <= iz0:
        return 0.0
    if a["geometry_kind"] == "box" and b["geometry_kind"] == "box":
        return (ix1 - ix0) * (iz1 - iz0)
    if a["geometry_kind"] == "polyshape" and b["geometry_kind"] == "box":
        clipped = clip_polygon(a["vertices"], b["bounds"])
    elif b["geometry_kind"] == "polyshape" and a["geometry_kind"] == "box":
        clipped = clip_polygon(b["vertices"], a["bounds"])
    else:
        # There is currently only one authoritative PolyShape.  Bounding-box
        # overlap is a conservative fallback if that changes.
        return (ix1 - ix0) * (iz1 - iz0)
    return polygon_area(clipped) if len(clipped) >= 3 else 0.0


def point_segment_distance(point, start, end) -> float:
    px, pz = point
    ax, az = start
    bx, bz = end
    dx, dz = bx - ax, bz - az
    denom = dx * dx + dz * dz
    if denom == 0:
        return math.hypot(px - ax, pz - az)
    t = max(0.0, min(1.0, ((px - ax) * dx + (pz - az) * dz) / denom))
    return math.hypot(px - (ax + t * dx), pz - (az + t * dz))


def polygons_distance(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    minimum = float("inf")
    for point in a:
        for i in range(len(b)):
            minimum = min(minimum, point_segment_distance(point, b[i], b[(i + 1) % len(b)]))
    for point in b:
        for i in range(len(a)):
            minimum = min(minimum, point_segment_distance(point, a[i], a[(i + 1) % len(a)]))
    return minimum


def load_baseline(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["zone_id"]: row for row in csv.DictReader(handle)}


def extract(scene: Path, baseline_path: Path) -> list[dict]:
    docs = parse_documents(scene.read_text(encoding="utf-8"))
    baseline = load_baseline(baseline_path)

    game_objects: dict[int, dict] = {}
    transforms: dict[int, dict] = {}
    colliders: dict[int, dict] = {}
    markers: dict[int, dict] = {}
    polyshapes: dict[int, dict] = {}
    component_to_go: dict[int, int] = {}

    for file_id, (class_id, body) in docs.items():
        if class_id == 1:
            game_objects[file_id] = {
                "name": field_text(body, "m_Name"),
                "components": parse_block_refs(body, "m_Component"),
            }
        elif class_id in (4, 224):
            go_id = field_ref(body, "m_GameObject")
            component_to_go[file_id] = go_id
            transforms[file_id] = {
                "go_id": go_id,
                "position": parse_vec(field_text(body, "m_LocalPosition")),
                "rotation": parse_vec(field_text(body, "m_LocalRotation"), 4),
                "scale": parse_vec(field_text(body, "m_LocalScale")),
                "father": field_ref(body, "m_Father"),
                "children": parse_block_refs(body, "m_Children"),
            }
        elif class_id == 65:
            go_id = field_ref(body, "m_GameObject")
            component_to_go[file_id] = go_id
            colliders[file_id] = {
                "go_id": go_id,
                "center": parse_vec(field_text(body, "m_Center")),
                "size": parse_vec(field_text(body, "m_Size")),
                "is_trigger": field_text(body, "m_IsTrigger", "0") == "1",
            }
        elif class_id == 114:
            go_id = field_ref(body, "m_GameObject")
            component_to_go[file_id] = go_id
            zone_match = re.search(r"^  zoneId: (.*)$", body, re.MULTILINE)
            if zone_match:
                markers[file_id] = {
                    "go_id": go_id,
                    "zone_id": zone_match.group(1).strip(),
                    "floor_id": field_text(body, "floorId"),
                    "zone_type_code": field_text(body, "zoneType"),
                }
            points = parse_points(body)
            if points:
                polyshapes[file_id] = {
                    "go_id": go_id,
                    "points": points,
                    "extrude": float(field_text(body, "m_Extrude", "0")),
                }

    go_transform = {}
    for transform_id, item in transforms.items():
        go_transform[item["go_id"]] = transform_id

    matrix_cache: dict[int, list[list[float]]] = {}

    def world_matrix(transform_id: int) -> list[list[float]]:
        if transform_id in matrix_cache:
            return matrix_cache[transform_id]
        item = transforms[transform_id]
        local = trs(item["position"], item["rotation"], item["scale"])
        result = matmul(world_matrix(item["father"]), local) if item["father"] in transforms else local
        matrix_cache[transform_id] = result
        return result

    descendants: dict[int, list[int]] = defaultdict(list)
    for transform_id, item in transforms.items():
        if item["father"]:
            descendants[item["father"]].append(transform_id)

    def descendant_transforms(transform_id: int) -> list[int]:
        result, queue = [], deque(descendants.get(transform_id, []))
        while queue:
            child = queue.popleft()
            result.append(child)
            queue.extend(descendants.get(child, []))
        return result

    zones: list[dict] = []
    for marker in markers.values():
        go_id = marker["go_id"]
        go = game_objects.get(go_id, {})
        name = go.get("name", "")
        if not name.startswith("SemanticZone__"):
            continue
        transform_id = go_transform.get(go_id)
        if not transform_id:
            continue
        base = baseline.get(marker["zone_id"], {})
        zone_type = base.get("zone_type", "Unknown")
        floor_id = marker["floor_id"] or base.get("floor_id", "")
        authoritative_poly = None
        authoritative_poly_transform = None
        for child_transform in descendant_transforms(transform_id):
            child_go_id = transforms[child_transform]["go_id"]
            child_go = game_objects.get(child_go_id, {})
            child_name = child_go.get("name", "")
            if not child_name.startswith(name + "_PolyShape"):
                continue
            for component_id in child_go.get("components", []):
                if component_id in polyshapes:
                    authoritative_poly = polyshapes[component_id]
                    authoritative_poly_transform = child_transform
                    break
            if authoritative_poly:
                break

        vertices: list[tuple[float, float]] = []
        y_values: list[float] = []
        geometry_kind = ""
        source_component_id = 0
        if authoritative_poly:
            geometry_kind = "polyshape"
            matrix = world_matrix(authoritative_poly_transform)
            for point in authoritative_poly["points"]:
                base_world = transform_point(matrix, point)
                top_world = transform_point(matrix, (point[0], point[1] + authoritative_poly["extrude"], point[2]))
                vertices.append((base_world[0], base_world[2]))
                y_values.extend((base_world[1], top_world[1]))
            source_component_id = next(
                component_id for component_id in game_objects[transforms[authoritative_poly_transform]["go_id"]]["components"]
                if component_id in polyshapes
            )
        else:
            collider_id = next((cid for cid in go.get("components", []) if cid in colliders), 0)
            if not collider_id:
                continue
            geometry_kind = "box"
            collider = colliders[collider_id]
            matrix = world_matrix(transform_id)
            cx, cy, cz = collider["center"]
            sx, sy, sz = collider["size"]
            local_footprint = [
                (cx - sx / 2, cy, cz - sz / 2),
                (cx + sx / 2, cy, cz - sz / 2),
                (cx + sx / 2, cy, cz + sz / 2),
                (cx - sx / 2, cy, cz + sz / 2),
            ]
            vertices = [(p[0], p[2]) for p in (transform_point(matrix, point) for point in local_footprint)]
            for x in (cx - sx / 2, cx + sx / 2):
                for y in (cy - sy / 2, cy + sy / 2):
                    for z in (cz - sz / 2, cz + sz / 2):
                        y_values.append(transform_point(matrix, (x, y, z))[1])
            source_component_id = collider_id

        if signed_area(vertices) < 0:
            vertices.reverse()
        xmin, xmax, zmin, zmax = bounds(vertices)
        centroid_x, centroid_z = polygon_centroid(vertices)
        y_min, y_max = min(y_values), max(y_values)
        zones.append({
            "zone_id": marker["zone_id"],
            "zone_type": zone_type,
            "zone_instance": base.get("zone_instance", ""),
            "floor_id": floor_id,
            "canonical_order": base.get("canonical_order", ""),
            "priority": base.get("priority", ""),
            "geometry_kind": geometry_kind,
            "center_x": centroid_x,
            "center_y": (y_min + y_max) / 2,
            "center_z": centroid_z,
            "size_x": xmax - xmin,
            "size_y": y_max - y_min,
            "size_z": zmax - zmin,
            "y_min": y_min,
            "y_max": y_max,
            "area_xz": polygon_area(vertices),
            "bounds": (xmin, xmax, zmin, zmax),
            "vertices": vertices,
            "source_game_object": name,
            "source_game_object_id": go_id,
            "source_transform_id": transform_id,
            "source_component_id": source_component_id,
            "confidence": base.get("confidence", ""),
            "evidence": base.get("evidence", ""),
            "notes": base.get("notes", ""),
        })
    return sorted(zones, key=lambda row: (float(row["canonical_order"] or 999), row["zone_id"]))


def priority_resolution(type_a: str, type_b: str) -> str:
    pair = {type_a, type_b}
    if pair == {"Stair", "StairEntrance"}:
        return "Stair wins; carve StairEntrance"
    if "DecisionZone" in pair and ("ExitCorridor" in pair or "BranchCorridor" in pair):
        return "DecisionZone wins"
    if pair == {"Exit", "BranchCorridor"}:
        return "Exit wins"
    return "REVIEW"


def audit(zones: list[dict], touch_tolerance: float = 0.03) -> dict:
    overlaps = []
    near_pairs = []
    orthogonality = []
    for zone in zones:
        errors = []
        points = zone["vertices"]
        for index in range(len(points)):
            dx = points[(index + 1) % len(points)][0] - points[index][0]
            dz = points[(index + 1) % len(points)][1] - points[index][1]
            error = min(abs(dx), abs(dz))
            if error > 0.002:
                errors.append({"edge": index + 1, "off_axis_m": error, "dx": dx, "dz": dz})
        if errors:
            orthogonality.append({"zone_id": zone["zone_id"], "errors": errors})

    for index, a in enumerate(zones):
        for b in zones[index + 1:]:
            vertical_overlap = min(a["y_max"], b["y_max"]) - max(a["y_min"], b["y_min"])
            if vertical_overlap <= 0.002:
                continue
            area = overlap_area(a, b)
            if area > 0.0001:
                overlaps.append({
                    "zone_a": a["zone_id"], "type_a": a["zone_type"],
                    "zone_b": b["zone_id"], "type_b": b["zone_type"],
                    "overlap_area_m2": area,
                    "vertical_overlap_m": vertical_overlap,
                    "resolution": priority_resolution(a["zone_type"], b["zone_type"]),
                })
            else:
                distance = polygons_distance(a["vertices"], b["vertices"])
                if distance <= 0.50:
                    near_pairs.append({
                        "zone_a": a["zone_id"], "zone_b": b["zone_id"],
                        "gap_m": distance,
                    })

    connectivity = {}
    for floor in ("Hall", "Platform"):
        members = [z for z in zones if z["zone_type"] != "Spawn" and floor in z["floor_id"]]
        graph = {z["zone_id"]: set() for z in members}
        for index, a in enumerate(members):
            for b in members[index + 1:]:
                if overlap_area(a, b) > 0.0001 or polygons_distance(a["vertices"], b["vertices"]) <= touch_tolerance:
                    graph[a["zone_id"]].add(b["zone_id"])
                    graph[b["zone_id"]].add(a["zone_id"])
        components, remaining = [], set(graph)
        while remaining:
            start = next(iter(remaining))
            component, queue = [], deque([start])
            remaining.remove(start)
            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in graph[current]:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        queue.append(neighbor)
            components.append(sorted(component))
        connectivity[floor] = sorted(components, key=lambda group: (-len(group), group))

    heights = defaultdict(list)
    for zone in zones:
        heights[zone["floor_id"]].append(zone["size_y"])

    return {
        "zone_count": len(zones),
        "polyshape_count": sum(z["geometry_kind"] == "polyshape" for z in zones),
        "overlaps": sorted(overlaps, key=lambda row: -row["overlap_area_m2"]),
        "near_pairs": sorted(near_pairs, key=lambda row: row["gap_m"]),
        "orthogonality": orthogonality,
        "connectivity": connectivity,
        "heights_by_floor": dict(heights),
    }


def write_csv(zones: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "zone_id", "zone_type", "zone_instance", "floor_id", "canonical_order", "priority",
        "geometry_kind", "center_x", "center_y", "center_z", "size_x", "size_y", "size_z",
        "y_min", "y_max", "area_xz", "vertices_xz", "source_game_object",
        "source_game_object_id", "source_transform_id", "source_component_id",
        "confidence", "evidence", "notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for zone in zones:
            row = {key: zone.get(key, "") for key in fieldnames}
            row["vertices_xz"] = ";".join(f"{x:.6f} {z:.6f}" for x, z in zone["vertices"])
            for key in ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z", "y_min", "y_max", "area_xz"):
                row[key] = f"{zone[key]:.6f}"
            writer.writerow(row)


def json_ready_zones(zones: list[dict]) -> list[dict]:
    ready = []
    for zone in zones:
        copy = dict(zone)
        copy["vertices"] = [[round(x, 6), round(z, 6)] for x, z in zone["vertices"]]
        copy["bounds"] = [round(value, 6) for value in zone["bounds"]]
        ready.append(copy)
    return ready


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    zones = extract(args.scene, args.baseline)
    report = audit(zones)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(zones, args.output_dir / "semantic_zones_source.csv")
    (args.output_dir / "semantic_zones_source.json").write_text(
        json.dumps(json_ready_zones(zones), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "semantic_zones_source_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "zones": len(zones),
        "polyshapes": report["polyshape_count"],
        "overlaps": len(report["overlaps"]),
        "review_overlaps": sum(item["resolution"] == "REVIEW" for item in report["overlaps"]),
        "orthogonality_issues": len(report["orthogonality"]),
        "output_dir": str(args.output_dir.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
