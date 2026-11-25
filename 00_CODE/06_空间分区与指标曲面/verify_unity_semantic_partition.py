#!/usr/bin/env python3
"""验证 Unity 场景中的语义区域是否与正式数据一致。

用途说明：读取已保存的 Unity 场景和正式 payload，逐区比较 GameObject 名称、
BoxCollider 数量、中心坐标、尺寸、Trigger 状态及可视化对象，并输出验证报告。
本工具只读取场景，不会修改 Unity。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from audit_unity_semantic_zones import (
    field_ref, field_text, identity, matmul, parse_block_refs, parse_documents,
    parse_vec, transform_point, trs,
)


def rounded_part(values):
    return tuple(round(float(value), 4) for value in values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    docs = parse_documents(args.scene.read_text(encoding="utf-8"))
    payload = json.loads(args.payload.read_text(encoding="utf-8"))["zones"]
    expected = {zone["zone_id"]: zone for zone in payload}

    game_objects, transforms, colliders, markers = {}, {}, {}, {}
    for file_id, (class_id, body) in docs.items():
        if class_id == 1:
            game_objects[file_id] = {
                "name": field_text(body, "m_Name"),
                "components": parse_block_refs(body, "m_Component"),
            }
        elif class_id in (4, 224):
            transforms[file_id] = {
                "go_id": field_ref(body, "m_GameObject"),
                "position": parse_vec(field_text(body, "m_LocalPosition")),
                "rotation": parse_vec(field_text(body, "m_LocalRotation"), 4),
                "scale": parse_vec(field_text(body, "m_LocalScale")),
                "father": field_ref(body, "m_Father"),
            }
        elif class_id == 65:
            colliders[file_id] = {
                "go_id": field_ref(body, "m_GameObject"),
                "center": parse_vec(field_text(body, "m_Center")),
                "size": parse_vec(field_text(body, "m_Size")),
                "is_trigger": field_text(body, "m_IsTrigger") == "1",
            }
        elif class_id == 114:
            match = re.search(r"^  zoneId: (.*)$", body, re.MULTILINE)
            if match:
                markers[field_ref(body, "m_GameObject")] = match.group(1).strip()

    go_transform = {item["go_id"]: transform_id for transform_id, item in transforms.items()}
    matrix_cache = {}
    def world_matrix(transform_id):
        if transform_id in matrix_cache:
            return matrix_cache[transform_id]
        item = transforms[transform_id]
        local = trs(item["position"], item["rotation"], item["scale"])
        matrix = matmul(world_matrix(item["father"]), local) if item["father"] in transforms else local
        matrix_cache[transform_id] = matrix
        return matrix

    actual = {}
    all_triggers = True
    name_mismatches = []
    for go_id, zone_id in markers.items():
        name = game_objects.get(go_id, {}).get("name", "")
        if not name.startswith("SemanticZone__"):
            continue
        expected_name = "SemanticZone__" + zone_id
        if name != expected_name:
            name_mismatches.append({"zone_id": zone_id, "actual_name": name, "expected_name": expected_name})
        matrix = world_matrix(go_transform[go_id])
        parts = []
        for component_id in game_objects[go_id]["components"]:
            if component_id not in colliders:
                continue
            collider = colliders[component_id]
            all_triggers = all_triggers and collider["is_trigger"]
            cx, cy, cz = collider["center"]
            sx, sy, sz = collider["size"]
            corners = [
                transform_point(matrix, (x, y, z))
                for x in (cx - sx / 2, cx + sx / 2)
                for y in (cy - sy / 2, cy + sy / 2)
                for z in (cz - sz / 2, cz + sz / 2)
            ]
            xs, ys, zs = zip(*corners)
            parts.append({
                "xmin": min(xs), "xmax": max(xs), "zmin": min(zs), "zmax": max(zs),
                "ymin": min(ys), "ymax": max(ys),
            })
        actual[zone_id] = parts

    mismatches = []
    for zone_id, zone in expected.items():
        expected_parts = sorted(
            rounded_part((p["xmin"], p["xmax"], p["zmin"], p["zmax"],
                          zone["center_y"] - zone["size_y"] / 2,
                          zone["center_y"] + zone["size_y"] / 2))
            for p in zone["collider_parts"]
        )
        actual_parts = sorted(
            rounded_part((p["xmin"], p["xmax"], p["zmin"], p["zmax"], p["ymin"], p["ymax"]))
            for p in actual.get(zone_id, [])
        )
        if expected_parts != actual_parts:
            mismatches.append({
                "zone_id": zone_id,
                "expected_count": len(expected_parts),
                "actual_count": len(actual_parts),
                "expected": expected_parts,
                "actual": actual_parts,
            })

    report = {
        "verified": not mismatches and not name_mismatches and all_triggers and len(actual) == 39,
        "zone_count": len(actual),
        "box_collider_count": sum(len(parts) for parts in actual.values()),
        "all_colliders_are_triggers": all_triggers,
        "generated_visual_count": sum(go["name"] == "__GeneratedSemanticZoneVisual" for go in game_objects.values()),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "name_mismatch_count": len(name_mismatches),
        "name_mismatches": name_mismatches,
        "scene": str(args.scene),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
