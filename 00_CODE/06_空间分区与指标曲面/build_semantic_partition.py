#!/usr/bin/env python3
"""生成正式、无重叠且相互连通的语义区域划分。

用途说明：这是“区域源数据 → 正式区域数据”的核心构建工具。它负责校正
横平竖直边界、统一楼层高度、按优先级挖空重合区域、对齐相邻边界，并把
复杂面域拆成 Unity 可使用的多个矩形 BoxCollider 分块。

This converts the manually edited Unity envelopes into a rectilinear partition.
Higher-priority zones remove area from lower-priority zones.  The two spawn
zones are kept separate and never expand Platform_Main.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


PLATFORM_FLOOR_Y = -15.771
HALL_FLOOR_Y = -11.8
HALL_CEILING_Y = -7.300001
HALL_CENTER_Y = (HALL_FLOOR_Y + HALL_CEILING_Y) / 2
HALL_HEIGHT = HALL_CEILING_Y - HALL_FLOOR_Y
PLATFORM_CENTER_Y = (PLATFORM_FLOOR_Y + HALL_FLOOR_Y) / 2
PLATFORM_HEIGHT = HALL_FLOOR_Y - PLATFORM_FLOOR_Y
EPS = 1e-7


def signed_area(points):
    return sum(points[i][0] * points[(i + 1) % len(points)][1]
               - points[(i + 1) % len(points)][0] * points[i][1]
               for i in range(len(points))) / 2


def bounds(points):
    xs, zs = zip(*points)
    return min(xs), max(xs), min(zs), max(zs)


def rect_vertices(xmin, xmax, zmin, zmax):
    return [[xmin, zmin], [xmax, zmin], [xmax, zmax], [xmin, zmax]]


def orthogonalize(points):
    """Replace each hand-drawn vertex by the intersection of averaged H/V edges."""
    n = len(points)
    orientations = []
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        orientations.append("V" if abs(b[0] - a[0]) < abs(b[1] - a[1]) else "H")
    if any(orientations[i] == orientations[(i + 1) % n] for i in range(n)):
        raise ValueError("PolyShape edges do not alternate horizontal/vertical")
    result = []
    for i in range(n):
        x = z = None
        for edge_index in ((i - 1) % n, i):
            a, b = points[edge_index], points[(edge_index + 1) % n]
            if orientations[edge_index] == "V":
                x = (a[0] + b[0]) / 2
            else:
                z = (a[1] + b[1]) / 2
        result.append([round(x, 6), round(z, 6)])
    return result


def point_in_polygon(x, z, polygon):
    inside = False
    j = len(polygon) - 1
    for i, (xi, zi) in enumerate(polygon):
        xj, zj = polygon[j]
        if (zi > z) != (zj > z):
            crossing_x = (xj - xi) * (z - zi) / (zj - zi) + xi
            if x < crossing_x:
                inside = not inside
        j = i
    return inside


def shares_layer(zone, other):
    if zone["zone_type"] == "Spawn" or other["zone_type"] == "Spawn":
        return zone["floor_id"] == other["floor_id"]
    a, b = zone["floor_id"], other["floor_id"]
    return ("Hall" in a and "Hall" in b) or ("Platform" in a and "Platform" in b)


def prepare_base_zones(zones):
    by_id = {zone["zone_id"]: zone for zone in zones}
    platform_xmin, platform_xmax, _, _ = by_id["Platform_Main"]["bounds"]
    hall_outside_xmin, hall_outside_xmax, _, _ = by_id["HallOutsideGate_Main"]["bounds"]
    gates = [zone for zone in zones if zone["zone_type"] == "Gate"]
    upper_gate = max(gates, key=lambda zone: (zone["bounds"][2] + zone["bounds"][3]) / 2)
    lower_gate = min(gates, key=lambda zone: (zone["bounds"][2] + zone["bounds"][3]) / 2)
    upper_bounds = upper_gate["bounds"]
    lower_bounds = lower_gate["bounds"]
    end_gate_ids = {upper_gate["zone_id"], lower_gate["zone_id"]}
    # The two physical end gates share their X edges regardless of numbering.
    end_gate_xmin = (upper_bounds[0] + lower_bounds[0]) / 2
    end_gate_xmax = (upper_bounds[1] + lower_bounds[1]) / 2
    result = []
    for source in zones:
        zone = dict(source)
        zone["vertices"] = [[float(x), float(z)] for x, z in source["vertices"]]
        if zone["zone_id"] == "HallInsideGate_Main":
            zone["vertices"] = orthogonalize(zone["vertices"])
            zone["vertices"] = [
                [
                    end_gate_xmax if x > 9.0 else end_gate_xmin if 1.0 < x < 4.0 else x,
                    upper_bounds[2] if z > 13.0 else lower_bounds[3] if z < -35.0 else z,
                ]
                for x, z in zone["vertices"]
            ]
            zone["geometry_kind"] = "orthogonal_polyshape_source"

        if zone["zone_id"] in end_gate_ids:
            _, _, zmin, zmax = bounds(zone["vertices"])
            zone["vertices"] = rect_vertices(end_gate_xmin, end_gate_xmax, zmin, zmax)
            zone["geometry_kind"] = "aligned_box_source"

        if zone["zone_id"] in {"StairEntrance_ABCD", "StairEntrance_EFGH"}:
            _, _, zmin, zmax = bounds(zone["vertices"])
            zone["vertices"] = rect_vertices(platform_xmin, platform_xmax, zmin, zmax)
            zone["geometry_kind"] = "aligned_box_source"

        # Expand each detached spawn only towards Platform_Main.  Its outside
        # edge and Z span stay fixed, while the inside edge becomes an exact
        # shared boundary with the platform (touching, never overlapping).
        if zone["zone_type"] == "Spawn":
            xmin, xmax, zmin, zmax = bounds(zone["vertices"])
            if (xmin + xmax) / 2 > 0:
                zone["vertices"] = rect_vertices(platform_xmax, xmax, zmin, zmax)
            else:
                zone["vertices"] = rect_vertices(xmin, platform_xmin, zmin, zmax)
            zone["geometry_kind"] = "aligned_box_source"

        if zone["zone_id"] in {"DecisionZone_ABCD", "DecisionZone_EFGH"}:
            _, _, zmin, zmax = bounds(zone["vertices"])
            zone["vertices"] = rect_vertices(hall_outside_xmin, hall_outside_xmax, zmin, zmax)
            zone["geometry_kind"] = "aligned_box_source"

        # The four outer Decision zones occupy the exact intersection of their
        # neighboring Branch and ExitCorridor envelopes.
        decision_alignment = {
            "DecisionZone_AB": ("BranchCorridor_A", "ExitCorridor_AB"),
            "DecisionZone_CD": ("BranchCorridor_C", "ExitCorridor_CD"),
            "DecisionZone_EF": ("BranchCorridor_E", "ExitCorridor_EF"),
            "DecisionZone_GH": ("BranchCorridor_G", "ExitCorridor_GH"),
        }
        if zone["zone_id"] in decision_alignment:
            branch = by_id[decision_alignment[zone["zone_id"]][0]]
            corridor = by_id[decision_alignment[zone["zone_id"]][1]]
            bx0, bx1, _, _ = branch["bounds"]
            _, _, cz0, cz1 = corridor["bounds"]
            zone["vertices"] = rect_vertices(bx0, bx1, cz0, cz1)
            zone["geometry_kind"] = "aligned_box_source"

        # Keep the central hall envelope only between the two central Decision
        # zones.  This eliminates thin end slivers and makes the shared edges exact.
        if zone["zone_id"] == "HallOutsideGate_Main":
            x0, x1, _, _ = bounds(zone["vertices"])
            south = by_id["DecisionZone_ABCD"]["bounds"]
            north = by_id["DecisionZone_EFGH"]["bounds"]
            zone["vertices"] = rect_vertices(x0, x1, south[3], north[2])
            zone["geometry_kind"] = "aligned_box_source"

        if zone["floor_id"] == "Hall":
            zone["center_y"], zone["size_y"] = HALL_CENTER_Y, HALL_HEIGHT
        elif zone["floor_id"] == "Platform":
            zone["center_y"], zone["size_y"] = PLATFORM_CENTER_Y, PLATFORM_HEIGHT
        # Cross-floor Stair volumes retain their manually confirmed vertical span.
        zone["y_min"] = zone["center_y"] - zone["size_y"] / 2
        zone["y_max"] = zone["center_y"] + zone["size_y"] / 2
        zone["bounds"] = list(bounds(zone["vertices"]))
        zone["priority"] = float(zone.get("priority") or 0)
        result.append(zone)
    return result


def unique_sorted(values):
    output = []
    for value in sorted(values):
        if not output or abs(value - output[-1]) > EPS:
            output.append(value)
    return output


def zone_cells(zone, all_zones, xs, zs):
    higher = [other for other in all_zones
              if other["priority"] > zone["priority"] and shares_layer(zone, other)]
    cells = set()
    for ix in range(len(xs) - 1):
        mx = (xs[ix] + xs[ix + 1]) / 2
        for iz in range(len(zs) - 1):
            mz = (zs[iz] + zs[iz + 1]) / 2
            if not point_in_polygon(mx, mz, zone["vertices"]):
                continue
            if any(point_in_polygon(mx, mz, other["vertices"]) for other in higher):
                continue
            cells.add((ix, iz))
    return cells


def merge_cells(cells, xs, zs):
    """Merge occupied grid cells into non-overlapping maximal rectangles."""
    row_runs = defaultdict(list)
    for iz in sorted({cell[1] for cell in cells}):
        row_xs = sorted(ix for ix, row in cells if row == iz)
        if not row_xs:
            continue
        start = previous = row_xs[0]
        for ix in row_xs[1:]:
            if ix != previous + 1:
                row_runs[iz].append((start, previous + 1))
                start = ix
            previous = ix
        row_runs[iz].append((start, previous + 1))

    active = {}
    rectangles = []
    for iz in range(len(zs) - 1):
        current = set(row_runs.get(iz, []))
        for run, start_iz in list(active.items()):
            if run not in current:
                rectangles.append((run[0], run[1], start_iz, iz))
                del active[run]
        for run in current:
            active.setdefault(run, iz)
    for run, start_iz in active.items():
        rectangles.append((run[0], run[1], start_iz, len(zs) - 1))
    return [
        {
            "xmin": xs[ix0], "xmax": xs[ix1],
            "zmin": zs[iz0], "zmax": zs[iz1],
        }
        for ix0, ix1, iz0, iz1 in rectangles
    ]


def boundary_loops(cells, xs, zs):
    edges = []
    occupied = cells
    for ix, iz in occupied:
        x0, x1, z0, z1 = xs[ix], xs[ix + 1], zs[iz], zs[iz + 1]
        if (ix, iz - 1) not in occupied:
            edges.append(((x0, z0), (x1, z0)))
        if (ix + 1, iz) not in occupied:
            edges.append(((x1, z0), (x1, z1)))
        if (ix, iz + 1) not in occupied:
            edges.append(((x1, z1), (x0, z1)))
        if (ix - 1, iz) not in occupied:
            edges.append(((x0, z1), (x0, z0)))
    outgoing = defaultdict(list)
    for start, end in edges:
        outgoing[start].append(end)
    loops = []
    unused = set(edges)
    while unused:
        start_edge = next(iter(unused))
        start, current = start_edge
        loop = [start, current]
        unused.remove(start_edge)
        while current != start:
            candidates = [end for end in outgoing[current] if (current, end) in unused]
            if not candidates:
                break
            # Prefer continuing straight, then turning left.  Point-touching
            # components are rare, but this keeps loops deterministic.
            previous = loop[-2]
            in_vec = (current[0] - previous[0], current[1] - previous[1])
            def turn_score(end):
                out_vec = (end[0] - current[0], end[1] - current[1])
                cross = in_vec[0] * out_vec[1] - in_vec[1] * out_vec[0]
                dot = in_vec[0] * out_vec[0] + in_vec[1] * out_vec[1]
                return (dot <= 0, -cross)
            end = sorted(candidates, key=turn_score)[0]
            unused.remove((current, end))
            loop.append(end)
            current = end
        if len(loop) >= 4 and loop[-1] == loop[0]:
            loops.append([[round(x, 6), round(z, 6)] for x, z in loop[:-1]])
    return loops


def build_partition(zones):
    base = prepare_base_zones(zones)
    xs = unique_sorted(x for zone in base for x, _ in zone["vertices"])
    zs = unique_sorted(z for zone in base for _, z in zone["vertices"])
    output = []
    for zone in base:
        cells = zone_cells(zone, base, xs, zs)
        if not cells:
            raise ValueError(f"Zone became empty: {zone['zone_id']}")
        rectangles = merge_cells(cells, xs, zs)
        loops = boundary_loops(cells, xs, zs)
        area = sum((rect["xmax"] - rect["xmin"]) * (rect["zmax"] - rect["zmin"])
                   for rect in rectangles)
        bx0 = min(rect["xmin"] for rect in rectangles)
        bx1 = max(rect["xmax"] for rect in rectangles)
        bz0 = min(rect["zmin"] for rect in rectangles)
        bz1 = max(rect["zmax"] for rect in rectangles)
        weighted_x = sum(((r["xmin"] + r["xmax"]) / 2) *
                         (r["xmax"] - r["xmin"]) * (r["zmax"] - r["zmin"])
                         for r in rectangles) / area
        weighted_z = sum(((r["zmin"] + r["zmax"]) / 2) *
                         (r["xmax"] - r["xmin"]) * (r["zmax"] - r["zmin"])
                         for r in rectangles) / area
        final = dict(zone)
        final.update({
            "geometry_kind": "orthogonal_composite" if len(rectangles) > 1 or len(loops) > 1 else "box",
            "center_x": weighted_x,
            "center_z": weighted_z,
            "size_x": bx1 - bx0,
            "size_z": bz1 - bz0,
            "area_xz": area,
            "bounds": [bx0, bx1, bz0, bz1],
            "boundary_loops": loops,
            "collider_parts": rectangles,
            "source_vertices": zone["vertices"],
        })
        output.append(final)
    return output


def audit_partition(zones):
    parts = []
    for zone in zones:
        for rect in zone["collider_parts"]:
            parts.append((zone, rect))
    overlaps = []
    def rectangles_touch(a, b):
        z_overlap = min(a["zmax"], b["zmax"]) - max(a["zmin"], b["zmin"])
        x_overlap = min(a["xmax"], b["xmax"]) - max(a["xmin"], b["xmin"])
        vertical_edge = (abs(a["xmax"] - b["xmin"]) <= EPS or abs(b["xmax"] - a["xmin"]) <= EPS) and z_overlap > EPS
        horizontal_edge = (abs(a["zmax"] - b["zmin"]) <= EPS or abs(b["zmax"] - a["zmin"]) <= EPS) and x_overlap > EPS
        return vertical_edge or horizontal_edge
    for i, (za, a) in enumerate(parts):
        for zb, b in parts[i + 1:]:
            if za["zone_id"] == zb["zone_id"] or not shares_layer(za, zb):
                continue
            ix = min(a["xmax"], b["xmax"]) - max(a["xmin"], b["xmin"])
            iz = min(a["zmax"], b["zmax"]) - max(a["zmin"], b["zmin"])
            if ix > EPS and iz > EPS:
                overlaps.append({"zone_a": za["zone_id"], "zone_b": zb["zone_id"], "area": ix * iz})
    decision_alignment = {}
    for zone_id in ("DecisionZone_AB", "DecisionZone_CD", "DecisionZone_EF", "DecisionZone_GH"):
        zone = next(item for item in zones if item["zone_id"] == zone_id)
        decision_alignment[zone_id] = zone["bounds"]
    internal_components = {}
    for zone in zones:
        rects = zone["collider_parts"]
        graph = {i: set() for i in range(len(rects))}
        for i, a in enumerate(rects):
            for j, b in enumerate(rects[i + 1:], i + 1):
                if rectangles_touch(a, b):
                    graph[i].add(j)
                    graph[j].add(i)
        remaining, count = set(graph), 0
        while remaining:
            count += 1
            stack = [remaining.pop()]
            while stack:
                current = stack.pop()
                for neighbor in graph[current]:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
        internal_components[zone["zone_id"]] = count

    layer_components = {}
    for layer in ("Hall", "Platform"):
        members = [zone for zone in zones
                   if zone["zone_type"] != "Spawn" and layer in zone["floor_id"]]
        graph = {zone["zone_id"]: set() for zone in members}
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                if any(rectangles_touch(ra, rb) for ra in a["collider_parts"] for rb in b["collider_parts"]):
                    graph[a["zone_id"]].add(b["zone_id"])
                    graph[b["zone_id"]].add(a["zone_id"])
        remaining, components = set(graph), []
        while remaining:
            start = remaining.pop()
            component, stack = [], [start]
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in graph[current]:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
            components.append(sorted(component))
        layer_components[layer] = sorted(components, key=lambda item: (-len(item), item))

    return {
        "zone_count": len(zones),
        "collider_part_count": sum(len(zone["collider_parts"]) for zone in zones),
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
        "decision_alignment_bounds": decision_alignment,
        "internally_disconnected_zones": {
            zone_id: count for zone_id, count in internal_components.items() if count != 1
        },
        "layer_connectivity_components": layer_components,
        "hall_height": HALL_HEIGHT,
        "platform_height": PLATFORM_HEIGHT,
    }


def write_zone_csv(zones, path):
    fields = [
        "zone_id", "zone_type", "zone_instance", "floor_id", "canonical_order", "priority",
        "geometry_kind", "center_x", "center_y", "center_z", "size_x", "size_y", "size_z",
        "y_min", "y_max", "area_xz", "boundary_loops_xz", "collider_part_count",
        "collider_parts_xz", "confidence", "evidence", "notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for zone in zones:
            row = {key: zone.get(key, "") for key in fields}
            for key in ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z", "y_min", "y_max", "area_xz"):
                row[key] = f"{float(zone[key]):.6f}"
            row["priority"] = f"{zone['priority']:.0f}"
            row["boundary_loops_xz"] = json.dumps(zone["boundary_loops"], ensure_ascii=False, separators=(",", ":"))
            row["collider_part_count"] = len(zone["collider_parts"])
            row["collider_parts_xz"] = json.dumps(zone["collider_parts"], ensure_ascii=False, separators=(",", ":"))
            writer.writerow(row)


def write_parts_csv(zones, path):
    fields = ["zone_id", "part_index", "center_x", "center_y", "center_z", "size_x", "size_y", "size_z"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for zone in zones:
            for index, rect in enumerate(zone["collider_parts"], 1):
                writer.writerow({
                    "zone_id": zone["zone_id"], "part_index": index,
                    "center_x": f"{(rect['xmin'] + rect['xmax']) / 2:.6f}",
                    "center_y": f"{zone['center_y']:.6f}",
                    "center_z": f"{(rect['zmin'] + rect['zmax']) / 2:.6f}",
                    "size_x": f"{rect['xmax'] - rect['xmin']:.6f}",
                    "size_y": f"{zone['size_y']:.6f}",
                    "size_z": f"{rect['zmax'] - rect['zmin']:.6f}",
                })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    current = json.loads(args.source.read_text(encoding="utf-8"))
    final = build_partition(current)
    report = audit_partition(final)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "semantic_zones.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "semantic_zones_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    unity_payload = {
        "zones": [
            {
                "zone_id": zone["zone_id"],
                "zone_type": zone["zone_type"],
                "center_y": zone["center_y"],
                "size_y": zone["size_y"],
                "collider_parts": zone["collider_parts"],
            }
            for zone in final
        ]
    }
    (args.output_dir / "semantic_zones_unity_payload.json").write_text(
        json.dumps(unity_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_zone_csv(final, args.output_dir / "semantic_zones.csv")
    write_parts_csv(final, args.output_dir / "semantic_zone_collider_parts.csv")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
