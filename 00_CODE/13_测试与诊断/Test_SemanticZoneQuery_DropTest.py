#!/usr/bin/env python3
"""站台层 / 站厅层语义分区掉针测试与散点分布图。

对 Platform(站台) 与 Hall(站厅) 分别指定垂直坐标 Y 进行掉针测试:

1. 楼层判定边界检查: 在关键 Y 值上验证 classify_floor 的 Platform/Hall 归属;
2. 随机掉针: 在楼层 XZ 包围盒内随机撒点, 调用 classify_point(x, y, z) 判定分区;
3. 合法性校验: 每个点的判定结果必须属于该楼层的合法分区集合(含跨层 Stair);
4. 分区中心命中检查: 每个分区首块 collider 中心点应命中自身或更高优先级分区;
5. 绘制散点分布图: 各分区散点按对应颜色着色, 叠加 collider 矩形边框。

用法:
    python Test_SemanticZoneQuery_DropTest.py [--n 15000] [--seed 20260821]
        [--platform-y 自动] [--hall-y 自动] [--out <图路径>]

脚本收于工程 TEST/ 目录。运行时会自动把工程根目录加入 sys.path, 以便导入根目录下的
SemanticZoneQuery; 分区几何默认取其 DEFAULT_GEOMETRY_PATH
(Results_Guidance_Spatial_Causal_Analysis/zone_annotation/semantic_zones.json)。
输出散点图默认保存在本脚本所在目录 (TEST/zone_droptest_platform_hall.png), 可从任意
工作目录启动。
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import numpy as np

# 本脚本收于工程 TEST/ 目录: 将工程根目录加入 sys.path, 才能导入根目录下的
# SemanticZoneQuery (分区几何与查询接口); __file__ 解析保证任意工作目录可运行。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from SemanticZoneQuery import DEFAULT_GEOMETRY_PATH, SemanticZoneQuery

plt.rcParams.update({
    "font.family": "sans-serif",
    # 英文字母/数字用 Times New Roman, 中文回退到微软雅黑/黑体。
    "font.sans-serif": ["Times New Roman", "Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "figure.dpi": 300,
    "font.size": 12,
})

NULL_COLOR = "#CCCCCC"

# 确定性色板: tab20 + tab20b + tab20c, 按 zone_id 排序分配, 保证每次运行一致。
_PALETTE = (
    list(plt.get_cmap("tab20").colors)
    + list(plt.get_cmap("tab20b").colors)
    + list(plt.get_cmap("tab20c").colors)
)


def build_zone_colors(query: SemanticZoneQuery) -> dict[str, str]:
    zone_ids = sorted({z.zone_id for z in query.zones})
    return {zid: _PALETTE[i % len(_PALETTE)] for i, zid in enumerate(zone_ids)}


def legal_zone_ids(query: SemanticZoneQuery, floor_name: str) -> set[str]:
    """该楼层掉针时允许出现的 zone_id 集合(含跨层 Stair)。"""
    return {z.zone_id for z in query.zones if z.belongs_to_floor(floor_name)}


def floor_xz_bounds(query: SemanticZoneQuery, floor_name: str, margin: float = 2.0):
    xs, zs = [], []
    for zone in query.zones:
        if not zone.belongs_to_floor(floor_name):
            continue
        for part in zone.collider_parts:
            xs += [part.xmin, part.xmax]
            zs += [part.zmin, part.zmax]
    return (min(xs) - margin, max(xs) + margin,
            min(zs) - margin, max(zs) + margin)


def print_floor_boundary_checks(query: SemanticZoneQuery) -> bool:
    """在关键 Y 值上验证楼层判定逻辑。"""
    print("[楼层判定边界检查]")
    ok = True
    checks = [
        (query.platform_floor_y - 0.1, None, "站台层底之下 -> 模型外"),
        (query.platform_floor_y, "Platform", "站台层底"),
        ((query.platform_floor_y + query.split_y) / 2.0, "Platform", "站台层中部"),
        (query.split_y - 1e-9, "Platform", "共享边界下方(紧贴)"),
        (query.split_y, "Hall", "共享边界 split_y 归属站厅"),
        ((query.split_y + query.hall_ceiling_y) / 2.0, "Hall", "站厅层中部"),
        (query.hall_ceiling_y, "Hall", "站厅层顶"),
        (query.hall_ceiling_y + 0.1, None, "站厅层顶之上 -> 模型外"),
    ]
    for yv, expect, desc in checks:
        got = query.classify_floor(yv)
        mark = "OK" if got == expect else "MISMATCH"
        if got != expect:
            ok = False
        print(f"  y={yv:+.4f}  ->  {str(got):<10s} 期望 {str(expect):<10s}  {desc:<22s} {mark}")
    return ok


def drop_test(query: SemanticZoneQuery, floor_name: str, y: float,
              n: int, rng: np.random.Generator):
    """随机掉针: 在楼层 XZ 包围盒内撒 n 个点并分类。"""
    xmin, xmax, zmin, zmax = floor_xz_bounds(query, floor_name)
    xs = rng.uniform(xmin, xmax, n)
    zs = rng.uniform(zmin, zmax, n)
    labels = [query.classify_point(float(x), float(y), float(z))
              for x, z in zip(xs, zs)]
    return xs, zs, labels, (xmin, xmax, zmin, zmax)


def zone_center_hit_check(query: SemanticZoneQuery, floor_name: str,
                          y: float, colors: dict[str, str]) -> bool:
    """每个分区首块 collider 中心点应命中自身或更高优先级分区。"""
    print(f"  [分区中心命中检查] floor={floor_name} y={y:.4f}")
    ok = True
    for zone in sorted(query.zones, key=lambda z: (-z.priority, z.zone_id)):
        if not zone.belongs_to_floor(floor_name):
            continue
        if not (zone.y_min <= y <= zone.y_max):
            continue
        part = zone.collider_parts[0]
        cx, cz = (part.xmin + part.xmax) / 2.0, (part.zmin + part.zmax) / 2.0
        got = query.classify_point(cx, y, cz)
        if got == zone.zone_id:
            flag = "命中自身"
        else:
            got_zone = next((z2 for z2 in query.zones if z2.zone_id == got), None)
            if got_zone is not None and got_zone.priority >= zone.priority:
                flag = f"被更高优先级覆盖: {got}(p{got_zone.priority:.0f})"
            else:
                flag = f"异常: {got}(p{got_zone.priority if got_zone else -1}) < 自身 p{zone.priority:.0f}"
                ok = False
        print(f"    {zone.zone_id:<26s} p={zone.priority:>4.0f} 中心=({cx:8.2f},{cz:8.2f}) "
              f"-> {got:<26s} {flag}")
    return ok


def draw_floor(ax, query: SemanticZoneQuery, floor_name: str, y: float,
               xs, zs, labels, bounds, colors: dict[str, str]) -> None:
    """绘制单楼层散点分布图。"""
    # 楼层 collider 矩形边框
    for zone in query.zones:
        if not zone.belongs_to_floor(floor_name):
            continue
        for part in zone.collider_parts:
            ax.add_patch(Rectangle(
                (part.xmin, part.zmin),
                part.xmax - part.xmin,
                part.zmax - part.zmin,
                facecolor="none", edgecolor="black", linewidth=1.2, zorder=2,
            ))
    # 分区名标注(首块 collider 中心)
    for zone in query.zones:
        if not zone.belongs_to_floor(floor_name):
            continue
        part = zone.collider_parts[0]
        cx, cz = (part.xmin + part.xmax) / 2.0, (part.zmin + part.zmax) / 2.0
        ax.text(cx, cz, zone.zone_id, ha="center", va="center", fontsize=5.5,
                color="black", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="gray", linewidth=0.4, alpha=0.75))

    # 分区着色散点
    point_colors = [colors.get(lbl, NULL_COLOR) for lbl in labels]
    ax.scatter(xs, zs, c=point_colors, s=2, alpha=0.55, zorder=3, rasterized=True)

    xmin, xmax, zmin, zmax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmin, zmax)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Z (m)", fontsize=12)
    ax.grid(True, alpha=0.25, linestyle=":")

    counts = Counter(labels)
    total = len(labels)
    nulls = counts.get("Null", 0)
    hit_rate = (total - nulls) / total * 100.0
    ax.set_title(
        f"{floor_name} 层掉针测试  (y = {y:.4f} m)\n"
        f"N={total}  命中={total - nulls}  未命中(Null)={nulls}  ({hit_rate:.1f}%)",
        fontsize=12, fontweight="bold",
    )


def print_statistics(floor_name: str, labels: list[str],
                     legal: set[str], colors: dict[str, str]) -> bool:
    print(f"[掉针统计] {floor_name}")
    counts = Counter(labels)
    total = len(labels)
    ok = True
    for lbl in sorted(counts, key=lambda k: (-counts[k], k)):
        c = counts[lbl]
        mark = ""
        if lbl == "Null":
            pass
        elif lbl not in legal:
            mark = "  <-- 非法分区!"
            ok = False
        print(f"  {lbl:<28s} {c:>6d}  ({c / total * 100:5.1f}%){mark}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=15000, help="每层掉针点数")
    parser.add_argument("--seed", type=int, default=20260821, help="随机种子")
    parser.add_argument("--platform-y", type=float, default=None,
                        help="站台层测试垂直坐标(默认取层中点)")
    parser.add_argument("--hall-y", type=float, default=None,
                        help="站厅层测试垂直坐标(默认取层中点)")
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY_PATH)
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent / "zone_droptest_platform_hall.png",
        help="散点分布图输出路径 (默认: 本脚本所在目录, 即 TEST/ 下)",
    )
    args = parser.parse_args()

    query = SemanticZoneQuery(args.geometry)
    colors = build_zone_colors(query)
    print(f"加载分区几何: {query.geometry_path}  (共 {len(query.zones)} 个分区)")

    # 1. 楼层判定边界检查
    all_ok = print_floor_boundary_checks(query)

    # 自动选择楼层中部测试高度
    platform_y = (query.platform_floor_y + query.split_y) / 2.0
    hall_y = (query.split_y + query.hall_ceiling_y) / 2.0
    platform_y = args.platform_y if args.platform_y is not None else platform_y
    hall_y = args.hall_y if args.hall_y is not None else hall_y

    rng = np.random.default_rng(args.seed)

    results = {}
    for floor_name, y in (("Platform", platform_y), ("Hall", hall_y)):
        print(f"\n===== {floor_name} 层掉针测试 @ y={y:.4f} =====")
        xs, zs, labels, bounds = drop_test(query, floor_name, y, args.n, rng)
        legal = legal_zone_ids(query, floor_name)
        all_ok &= print_statistics(floor_name, labels, legal, colors)
        all_ok &= zone_center_hit_check(query, floor_name, y, colors)
        results[floor_name] = (xs, zs, labels, bounds)

    # 2. 散点分布图
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, floor_name, y in zip(axes, ("Platform", "Hall"), (platform_y, hall_y)):
        xs, zs, labels, bounds = results[floor_name]
        draw_floor(ax, query, floor_name, y, xs, zs, labels, bounds, colors)

    # 统一图例
    seen = set()
    handles, names = [], []
    for floor_name in ("Platform", "Hall"):
        _, _, labels, _ = results[floor_name]
        for lbl in sorted(set(labels)):
            if lbl in seen:
                continue
            seen.add(lbl)
            clr = colors.get(lbl, NULL_COLOR)
            handles.append(Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=clr, markersize=7, label=lbl))
            names.append(lbl)
    fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=6, fontsize=7, framealpha=0.9, title="分区 (zone_id)")

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n散点分布图已保存: {out_path}")

    if not all_ok:
        print("[FAIL] 检测到异常, 请检查上方输出。")
        sys.exit(1)
    print("[PASS] 所有楼层判定逻辑检查通过。")


if __name__ == "__main__":
    main()
