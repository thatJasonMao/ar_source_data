"""
眼动向量与 Camera 正前方 0.7m 垂直平面的交点热区分析

逻辑：
  - 复用 Extract_GazeSphereHeatmap 的 3D 相对向量计算逻辑
  - Camera 位于 Agent/Subject 根节点局部 (0, 1.771, 0.2) m
  - Canvas 位于 Agent 根节点局部 z=0.9 m，因此从 Camera 起算的
    前向投影距离为 d=0.7 m
  - 视线起点取 Camera，而不是 Agent/Subject 根节点；垂直轴为穿点距地面高度
  - 在平面上做 KDE 热力图，呈现第一人称视角的眼动分布
  - 峰值处画标记并标注坐标（文字不越出既有坐标范围）
  - 按场景类别叠加 Unity World Space Canvas 面域线框：Full 两类全部、
    MiniMap 仅 MiniMap、TextBar 仅 TextBar；ArrowRibbon/StaticSign 不叠加

输出：每个场景目录一张热力图 PNG（无子图、无 HTML）
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from common import PROJECT_ROOT, GAZE_TOPO_DIR, SCENARIO_FOLDERS

# 输入/输出目录
RELATIVE_POS_BASE_DIR = os.path.join(PROJECT_ROOT, "Results_RelativePosition")
RESULTS_BASE_DIR = os.path.join(GAZE_TOPO_DIR, "Results_GazePlaneHeatmap")

# 香农熵汇总表（由 Extract_GazeSphereHeatmap.py 预先算好，本脚本只读）
ENTROPY_CSV = os.path.join(GAZE_TOPO_DIR, "Results_GazeSphereHeatmap",
                           "ShannonEntropy_Summary.csv")

# Camera 到 Canvas 垂直平面的前向距离（单位: m）。Canvas 的 Agent
# 根节点局部 z 仍为 0.9 m；Camera 局部 z=0.2 m，所以二者相差 0.7 m。
PLANE_DISTANCE = 0.7
CAMERA_LOCAL_Z_M = 0.2
CANVAS_PLANE_LOCAL_Z_M = CAMERA_LOCAL_Z_M + PLANE_DISTANCE

# Camera相对脚底（SubjectWorld/rig root）的局部Y高度（单位: m）。
EYE_HEIGHT_M = 1.771

# 热力图绘制范围 (X 方向: 左右, Z 方向: 上下 = 穿点距地面高度)
# 下限 1.0 m 盖住 MiniMap 底边 1.2256 m；上限 2.5 m 高于 TextBar 最高点
# 2.0214 m（超出 2.5 m 的向上注视尾迹会被裁剪）。
X_RANGE = (-1.0, 1.0)
Z_RANGE = (1.0, 2.5)

# 坐标轴刻度步长 (两轴统一)
TICK_STEP = 0.25

# KDE 网格分辨率
GRID_POINTS = 200

# 峰值标注样式：在密度最大值处画标记并标注坐标
PEAK_FONT_SIZE = 16.5   # 标注坐标文字字号 = 原 11pt 的 150%
PEAK_LABEL_DX = 0.30    # 文字与峰值的水平间距 (m)
PEAK_LABEL_DY = 0.05    # 文字与峰值的垂直间距 (m)

# Unity World Space Canvas 几何。AgentNEW.prefab 的实际比例为
# 0.0008 m/UI unit，与 Reconstruct_Minimap_Text_Gaze_AOI.py 保持一致。
CANVAS_SCALE_M_PER_UI_UNIT = 0.0008
UI_REGION_SPECS = (
    {
        "name": "MiniMap",
        "canvas_xy_m": (0.164, 1.668),
        "anchored_xy_ui": (132.0, -153.0),
        "size_ui": (400.0, 800.0),
        "color": "#00A6D6",
    },
    {
        "name": "TextBarDirection",
        "canvas_xy_m": (0.0, 1.699),
        "anchored_xy_ui": (0.5, 353.0),
        "size_ui": (1000.0, 100.0),
        "color": "#E66100",
    },
    {
        "name": "TextBarDistance",
        "canvas_xy_m": (0.0, 1.699),
        "anchored_xy_ui": (0.5, 254.0),
        "size_ui": (1000.0, 100.0),
        "color": "#E66100",
    },
)

# 各场景类别需叠加的面域线框（按场景名前缀匹配，顺序保持 UI_REGION_SPECS）
#   Full: 两类全部;  MiniMap: 仅 MiniMap;  TextBar: 仅 TextBar 类;
#   ArrowRibbon / StaticSign: 不叠加任何面域
SCENARIO_UI_NAMES = {
    "Full": ("MiniMap", "TextBarDirection", "TextBarDistance"),
    "MiniMap": ("MiniMap",),
    "TextBar": ("TextBarDirection", "TextBarDistance"),
    "ArrowRibbon": (),
    "StaticSign": (),
}


# 显示名称统一（仅作用于图面显示文本）：Full → Multimodal、ArrowRibbon → Arrow。
# 内部键名、目录名、文件名与路径均保持原样。
SCENARIO_DISPLAY_NAMES = {
    "Full": "Multimodal",
    "ArrowRibbon": "Arrow",
}


def display_scenario_name(scenario_name):
    """返回图面显示用的场景名（只替换类别前缀，例如 Full-LowSmoke → Multimodal-LowSmoke）。"""
    prefix, sep, rest = scenario_name.partition("-")
    return SCENARIO_DISPLAY_NAMES.get(prefix, prefix) + sep + rest


_ENTROPY_CACHE = None


def load_entropy_map():
    """读取香农熵汇总表，返回 {场景名: 熵值(bit)} 字典（缓存，只读一次）。

    CSV 由 Extract_GazeSphereHeatmap.py 生成，两列：场景, 香农熵(bit)，
    场景名为内部键名（如 Full-LowSmoke），带 UTF-8 BOM。
    """
    global _ENTROPY_CACHE
    if _ENTROPY_CACHE is None:
        _ENTROPY_CACHE = {}
        if os.path.exists(ENTROPY_CSV):
            df = pd.read_csv(ENTROPY_CSV, encoding="utf-8-sig")
            for _, row in df.iterrows():
                _ENTROPY_CACHE[str(row.iloc[0]).strip()] = float(row.iloc[1])
        else:
            print(f"  [警告] 找不到香农熵汇总表: {ENTROPY_CSV}")
    return _ENTROPY_CACHE


def annotate_shannon_entropy(ax, scenario_name):
    """在坐标轴范围内正中间、贴着最上侧边缘标注香农熵（保留五位小数）。

    字号与质心（峰值坐标）标注看齐（PEAK_FONT_SIZE）。场景无熵值时跳过。
    """
    entropy = load_entropy_map().get(scenario_name)
    if entropy is None or not np.isfinite(entropy):
        return
    x_center = (X_RANGE[0] + X_RANGE[1]) / 2.0
    ax.text(x_center, Z_RANGE[1] - 0.05, f"H = {entropy:.5f} bit",
            ha="center", va="top", fontsize=PEAK_FONT_SIZE,
            zorder=8)


def ui_specs_for_scenario(scenario_name):
    """返回该场景需要叠加线框的面域 spec 列表（无匹配前缀时为空列表）。"""
    prefix = scenario_name.split("-")[0]
    allowed = SCENARIO_UI_NAMES.get(prefix, ())
    return [s for s in UI_REGION_SPECS if s["name"] in allowed]


def unity_ui_rect_to_plane_bounds(spec):
    """Convert a centered Unity RectTransform from UI units to plane metres."""
    canvas_x, canvas_y = spec["canvas_xy_m"]
    anchored_x, anchored_y = spec["anchored_xy_ui"]
    width_ui, height_ui = spec["size_ui"]

    center_x = canvas_x + anchored_x * CANVAS_SCALE_M_PER_UI_UNIT
    center_y = canvas_y + anchored_y * CANVAS_SCALE_M_PER_UI_UNIT
    half_width = width_ui * CANVAS_SCALE_M_PER_UI_UNIT / 2.0
    half_height = height_ui * CANVAS_SCALE_M_PER_UI_UNIT / 2.0
    return (
        center_x - half_width,
        center_x + half_width,
        center_y - half_height,
        center_y + half_height,
    )


def draw_ui_region_annotations(ax, specs=None):
    """按面域清单绘制 Unity UI 矩形线框与英文标签（MiniMap / TextBar）。

    specs 为 None 时绘制全部面域；标签按类聚合：MiniMap 单独一个，
    TextBarDirection/TextBarDistance 合并标一个 "TextBar"。
    """
    if specs is None:
        specs = UI_REGION_SPECS

    bounds_by_name = {}
    for spec in specs:
        x_min, x_max, y_min, y_max = unity_ui_rect_to_plane_bounds(spec)
        bounds_by_name[spec["name"]] = (x_min, x_max, y_min, y_max)
        ax.add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                fill=False,
                edgecolor=spec["color"],
                linewidth=2.0,
                linestyle="-",
                zorder=5,
            )
        )

    label_style = {
        "va": "center",
        "fontsize": 11,
        "fontweight": "bold",
        "color": "black",
        "zorder": 6,
        "bbox": None,
    }

    if "MiniMap" in bounds_by_name:
        mini = bounds_by_name["MiniMap"]
        ax.text(mini[1] + 0.02, (mini[2] + mini[3]) / 2.0,
                "MiniMap", ha="left", **label_style)

    text_bounds = [b for n, b in bounds_by_name.items()
                   if n.startswith("TextBar")]
    if text_bounds:
        ax.text(
            max(b[1] for b in text_bounds) + 0.03,
            (min(b[2] for b in text_bounds)
             + max(b[3] for b in text_bounds)) / 2.0,
            "TextBar", ha="left", **label_style
        )


def annotate_density_peak(ax, peak_x, peak_y):
    """在密度峰值处画标记并标注坐标（字号 PEAK_FONT_SIZE）。

    文字优先放在峰值左上方（与既有布局一致），空间不足时依次尝试
    右上/左下/右下，间距不足则缩小间距；保证文字包框始终落在既有
    X_RANGE / Z_RANGE 范围内、不越界。
    """
    text = f"({peak_x:.2f}, {peak_y:.2f})"
    # 文字包框尺寸的保守估算（数据单位，对应 PEAK_FONT_SIZE=16.5）
    text_w, text_h = 0.55, 0.13

    peak_color = "#800080"
    ax.plot(peak_x, peak_y, marker="^", markersize=5, linestyle="none",
            color=peak_color, zorder=8, clip_on=False)

    placed = False
    for scale in (1.0, 0.5, 0.0):            # 间距：充足 → 减半 → 贴边
        for sx, sy, ha, va in (
            (-1, 1, "right", "bottom"),      # 左上
            (1, 1, "left", "bottom"),        # 右上
            (-1, -1, "right", "top"),        # 左下
            (1, -1, "left", "top"),          # 右下
        ):
            label_x = peak_x + sx * PEAK_LABEL_DX * scale
            label_y = peak_y + sy * PEAK_LABEL_DY * scale
            if ((ha == "right" and label_x - text_w < X_RANGE[0])
                    or (ha == "left" and label_x + text_w > X_RANGE[1])):
                continue
            if ((va == "top" and label_y - text_h < Z_RANGE[0])
                    or (va == "bottom" and label_y + text_h > Z_RANGE[1])):
                continue
            placed = True
            break
        if placed:
            break

    if not placed:  # 兜底：锚点硬钳制在范围内（正常数据几乎不会走到）
        label_x = float(np.clip(peak_x, X_RANGE[0] + text_w, X_RANGE[1] - text_w))
        label_y = float(np.clip(peak_y, Z_RANGE[0] + text_h, Z_RANGE[1] - text_h))
        ha, va = "center", "center"

    ax.annotate(
        text,
        xy=(peak_x, peak_y),
        xytext=(label_x, label_y),
        textcoords="data",
        ha=ha, va=va,
        fontsize=PEAK_FONT_SIZE, color=peak_color,
        bbox=None,
        arrowprops={"arrowstyle": "-", "color": peak_color,
                    "linewidth": 0.8, "shrinkA": 2, "shrinkB": 4},
        zorder=8, annotation_clip=False,
    )


def dry_run():
    """不读取试次数据，校验：面域几何、按场景类别的面域叠加数量与标签、
    峰值标注文字锚点是否始终落在坐标范围内。"""
    expected = {
        "Full": (3, ["MiniMap", "TextBar"]),
        "MiniMap": (1, ["MiniMap"]),
        "TextBar": (2, ["TextBar"]),
        "ArrowRibbon": (0, []),
        "StaticSign": (0, []),
    }
    for prefix, (n_patch, labels) in expected.items():
        fig = matplotlib.figure.Figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        ax.set_xlim(X_RANGE)
        ax.set_ylim(Z_RANGE)
        specs = ui_specs_for_scenario(f"{prefix}-LowSmoke")
        draw_ui_region_annotations(ax, specs)

        if len(ax.patches) != n_patch:
            raise RuntimeError(
                f"Dry run failed: {prefix} expects {n_patch} outline(s), "
                f"got {len(ax.patches)}."
            )
        if [t.get_text() for t in ax.texts] != labels:
            raise RuntimeError(f"Dry run failed: {prefix} labels unexpected.")

        for spec in specs:
            x_min, x_max, y_min, y_max = unity_ui_rect_to_plane_bounds(spec)
            if not (X_RANGE[0] <= x_min < x_max <= X_RANGE[1]
                    and Z_RANGE[0] <= y_min < y_max <= Z_RANGE[1]):
                raise RuntimeError(f"Dry run failed: {spec['name']} is outside plot bounds.")
        fig.clear()

    # 峰值标注：中心 / 左上 / 右上角三个位置，文字锚点均不得越出坐标范围
    for px, py in ((0.0, 1.75), (-0.8, 2.35), (0.85, 2.42)):
        fig = matplotlib.figure.Figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        ax.set_xlim(X_RANGE)
        ax.set_ylim(Z_RANGE)
        annotate_density_peak(ax, px, py)
        tx, ty = ax.texts[-1].get_position()
        if not (X_RANGE[0] < tx < X_RANGE[1]
                and Z_RANGE[0] < ty < Z_RANGE[1]):
            raise RuntimeError(
                f"Dry run failed: peak label anchor out of range "
                f"({tx:.3f}, {ty:.3f})."
            )
        fig.clear()

    print(
        "Dry run passed. Unity UI bounds on the Camera-relative 0.7 m "
        "plane / Agent-root local z=0.9 m (Full):"
    )
    for spec in UI_REGION_SPECS:
        x_min, x_max, y_min, y_max = unity_ui_rect_to_plane_bounds(spec)
        print(
            f"  {spec['name']}: x=[{x_min:.4f}, {x_max:.4f}] m, "
            f"vertical=[{y_min:.4f}, {y_max:.4f}] m"
        )
    print("Scenario UI overlays: Full -> all; MiniMap -> MiniMap only; "
          "TextBar -> TextBar only; ArrowRibbon/StaticSign -> none.")


def get_custom_cmap():
    """
    构造自定义颜色映射：
    - 密度趋近 0 时，显示透明的淡灰色
    - 密度大于阈值时，平滑过渡到 jet 热力图颜色
    """
    jet = cm.get_cmap('jet')
    colors = jet(np.linspace(0, 1, 256))
    base_color = np.array([0.8, 0.8, 0.8, 0.0])
    threshold_idx = 30
    for i in range(256):
        if i < threshold_idx:
            alpha = i / float(threshold_idx)
            jet_color = colors[i].copy()
            jet_color[3] = 0.85
            colors[i] = base_color * (1 - alpha) + jet_color * alpha
        else:
            colors[i][3] = 0.85
    return LinearSegmentedColormap.from_list('custom_plane', colors)


def calculate_3d_relative_vectors(df):
    """
    与 Extract_GazeSphereHeatmap 完全一致的三维相对向量计算。
    """
    sub_x = df['SubjectWorldX'].to_numpy(dtype=float)
    sub_y = df['SubjectWorldY'].to_numpy(dtype=float)
    sub_z = df['SubjectWorldZ'].to_numpy(dtype=float)
    sub_rot_y = df['SubjectRotY_deg'].to_numpy(dtype=float)

    gaze_x = df['GazeWorldX'].to_numpy(dtype=float)
    gaze_y = df['GazeWorldY'].to_numpy(dtype=float)
    gaze_z = df['GazeWorldZ'].to_numpy(dtype=float)

    dx = gaze_x - sub_x
    dy = gaze_y - sub_y
    dz = gaze_z - sub_z

    rotation_rad = np.radians(sub_rot_y)
    cos_rad = np.cos(rotation_rad)
    sin_rad = np.sin(rotation_rad)

    local_x = cos_rad * dx - sin_rad * dz
    local_z = sin_rad * dx + cos_rad * dz
    local_y = dy

    return np.column_stack([local_x, local_z, local_y])


def project_to_plane(points_3d):
    """
    将三维点从 Camera 原点投射到正前方 PLANE_DISTANCE=0.7 m 的平面。

    calculate_3d_relative_vectors 返回 (local_x, local_z, local_y):
      local_x = 左右, local_z = 前方, local_y = 注视目标相对脚底的高度

    数据中 Subject 位置是脚底/rig root。Camera 相对该根节点的局部位置为
    (0, EYE_HEIGHT_M, CAMERA_LOCAL_Z_M)，因此先把目标点转换为相对 Camera
    的方向：
      camera_x = local_x
      camera_forward = local_z - CAMERA_LOCAL_Z_M
      camera_y = local_y - EYE_HEIGHT_M

    与 Camera 前方 PLANE_DISTANCE 的平面相交时：
      t = PLANE_DISTANCE / camera_forward
      X_plane = camera_x * t
      Vertical_plane = EYE_HEIGHT_M + camera_y * t

    该平面等价于 Agent 根节点局部 z=0.9 m。

    仅保留 camera_forward > 0（从 Camera 指向前方）的数据。

    返回: (N, 2) array [[x_plane, z_plane], ...]
    """
    vx = points_3d[:, 0]
    vz = points_3d[:, 1] - CAMERA_LOCAL_Z_M
    vy = points_3d[:, 2] - EYE_HEIGHT_M
    valid = np.isfinite(points_3d).all(axis=1) & (vz > 1e-5)
    if not np.any(valid):
        return np.empty((0, 2))

    vx_v, vz_v, vy_v = vx[valid], vz[valid], vy[valid]
    t = PLANE_DISTANCE / vz_v
    x_plane = vx_v * t            # 水平偏移（左右）
    # Camera局部Y为眼高，输出仍为穿点距地面的绝对高度。
    z_plane = EYE_HEIGHT_M + vy_v * t
    return np.column_stack([x_plane, z_plane])


def draw_plane_heatmap(points_2d, scenario_name, output_dir, pbar=None):
    """
    在垂直平面上绘制 KDE 热力图
    """
    if pbar:
        pbar.set_postfix_str("Project vectors")
    points_2d = points_2d[np.all(np.isfinite(points_2d), axis=1)]
    if len(points_2d) < 10:
        if pbar:
            pbar.set_postfix_str("Skipped: too few points")
            pbar.update(5)
        return

    # 裁剪到绘图范围
    in_range = (
        (points_2d[:, 0] >= X_RANGE[0]) & (points_2d[:, 0] <= X_RANGE[1]) &
        (points_2d[:, 1] >= Z_RANGE[0]) & (points_2d[:, 1] <= Z_RANGE[1])
    )
    points_clipped = points_2d[in_range]
    if len(points_clipped) < 10:
        if pbar:
            pbar.set_postfix_str("Skipped: too few in-range")
            pbar.update(5)
        return

    if pbar:
        pbar.update(1)
        pbar.set_postfix_str("KDE")

    # ── KDE 密度估计 ──
    try:
        kde = gaussian_kde(points_clipped.T, bw_method='scott')
    except Exception:
        if pbar:
            pbar.set_postfix_str("KDE failed")
            pbar.update(4)
        return

    xx, zz = np.mgrid[X_RANGE[0]:X_RANGE[1]:complex(GRID_POINTS),
                       Z_RANGE[0]:Z_RANGE[1]:complex(GRID_POINTS)]
    positions = np.vstack([xx.ravel(), zz.ravel()])
    density = np.reshape(kde(positions).T, xx.shape)

    # 归一化
    d_min, d_max = density.min(), density.max()
    if d_max - d_min > 1e-8:
        density_norm = (density - d_min) / (d_max - d_min)
    else:
        density_norm = np.zeros_like(density)

    if pbar:
        pbar.update(1)
        pbar.set_postfix_str("Plotting")

    # ── 绘图 ──
    x_span = X_RANGE[1] - X_RANGE[0]
    z_span = Z_RANGE[1] - Z_RANGE[0]
    fig_w = 8
    fig_h = fig_w * z_span / x_span

    custom_cmap = get_custom_cmap()
    fig = matplotlib.figure.Figure(figsize=(fig_w, fig_h), dpi=300)
    ax = fig.add_subplot(111)

    extent = [X_RANGE[0], X_RANGE[1], Z_RANGE[0], Z_RANGE[1]]
    im = ax.imshow(
        density_norm.T, extent=extent, origin='lower',
        cmap=custom_cmap, aspect='equal'
    )

    # 统一刻度步长
    x_ticks = np.arange(X_RANGE[0], X_RANGE[1] + 1e-9, TICK_STEP)
    z_ticks = np.arange(Z_RANGE[0], Z_RANGE[1] + 1e-9, TICK_STEP)
    ax.set_xticks(x_ticks)
    ax.set_yticks(z_ticks)

    # 浅灰色背景网格
    ax.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.7)

    ax.set_xlim(X_RANGE)
    ax.set_ylim(Z_RANGE)
    ax.set_xlabel('Horizontal (m)', fontsize=14)
    ax.set_ylabel('Vertical (m)', fontsize=14)
    ax.set_title(display_scenario_name(scenario_name), fontsize=16, y=1.02)

    # Mark the maximum on the existing density grid; Y is the vertical coordinate.
    peak_index = np.unravel_index(np.argmax(density), density.shape)
    peak_x = float(xx[peak_index])
    peak_y = float(zz[peak_index])

    # 峰值标记与坐标标注（文字自动避让轴边界，字号见 PEAK_FONT_SIZE）
    annotate_density_peak(ax, peak_x, peak_y)

    # 香农熵标注（读自 Extract_GazeSphereHeatmap 的汇总表，顶部居中）
    annotate_shannon_entropy(ax, scenario_name)

    # Unity UI regions（按场景类别选择性叠加，见 SCENARIO_UI_NAMES）
    draw_ui_region_annotations(ax, ui_specs_for_scenario(scenario_name))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Relative density', fontsize=13)

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{scenario_name}_PlaneHeatmap.png")

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig)
    fig.savefig(out_path, dpi=300, transparent=False, bbox_inches="tight")
    fig.clear()

    if pbar:
        pbar.update(1)
        pbar.set_postfix_str("Done")


def process_scenario(scenario_folder, position=0):
    """单个场景：读取数据 → 计算向量 → 平面投影 → 热力图"""
    if tqdm is None:
        print(f"  [开始] 组别: {scenario_folder}")
        pbar = None
    else:
        pbar = tqdm(total=3, desc=f"{scenario_folder[:20]:<20}",
                    position=position, leave=True)

    input_csv = os.path.join(RELATIVE_POS_BASE_DIR, scenario_folder,
                             "AllSubjects_RelativePosInfo.csv")
    if not os.path.exists(input_csv):
        msg = f"  [跳过] {scenario_folder} 找不到数据文件"
        if pbar:
            pbar.set_postfix_str("Skipped: no data"); pbar.update(3); pbar.close()
        return msg

    try:
        df = pd.read_csv(input_csv)
    except Exception:
        msg = f"  [错误] {scenario_folder} 无法读取 CSV"
        if pbar:
            pbar.set_postfix_str("Error: read failed"); pbar.update(3); pbar.close()
        return msg

    required_cols = ['SubjectWorldX', 'SubjectWorldY', 'SubjectWorldZ',
                     'SubjectRotY_deg', 'GazeWorldX', 'GazeWorldY', 'GazeWorldZ']
    if not all(c in df.columns for c in required_cols):
        msg = f"  [跳过] {scenario_folder} 列不完整"
        if pbar:
            pbar.set_postfix_str("Skipped: missing cols"); pbar.update(3); pbar.close()
        return msg

    df = df.dropna(subset=required_cols)
    if len(df) == 0:
        msg = f"  [跳过] {scenario_folder} 无有效数据"
        if pbar:
            pbar.set_postfix_str("Skipped: empty"); pbar.update(3); pbar.close()
        return msg

    points_3d = calculate_3d_relative_vectors(df)
    points_2d = project_to_plane(points_3d)

    draw_plane_heatmap(points_2d, scenario_folder, RESULTS_BASE_DIR, pbar=pbar)

    if pbar:
        pbar.close()
    return f"  [完成] 组别: {scenario_folder}"


def main():
    print(f"\n{'=' * 70}")
    print("眼动向量平面交点热区分析 (多线程)")
    print(f"{'=' * 70}")
    print(f"Camera到平面前向距离: {PLANE_DISTANCE} m")
    print(f"Camera局部位置: (0, {EYE_HEIGHT_M}, {CAMERA_LOCAL_Z_M}) m")
    print(f"Canvas平面Agent根节点局部Z: {CANVAS_PLANE_LOCAL_Z_M} m")
    print(f"结果输出目录: {RESULTS_BASE_DIR}")
    print(f"场景数量: {len(SCENARIO_FOLDERS)}")
    print(f"{'=' * 70}\n")

    os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_scenario = {}
        for idx, scenario_folder in enumerate(SCENARIO_FOLDERS):
            future = executor.submit(process_scenario, scenario_folder, idx)
            future_to_scenario[future] = scenario_folder

        for future in as_completed(future_to_scenario):
            try:
                future.result()
            except Exception as exc:
                scenario_folder = future_to_scenario[future]
                if tqdm is not None:
                    tqdm.write(f"  [异常] 组别 {scenario_folder} 产生错误: {exc}")
                else:
                    print(f"  [异常] 组别 {scenario_folder} 产生错误: {exc}")

    print(f"\n{'=' * 70}")
    print("全部处理完成！")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draw gaze-plane heatmaps with Unity UI region outlines."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate UI geometry and plotting objects without reading trial data.",
    )
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    else:
        main()
