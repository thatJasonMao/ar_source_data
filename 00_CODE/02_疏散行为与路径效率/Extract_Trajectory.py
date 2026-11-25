import os
import json
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from common import PROJECT_ROOT, RAW_DATA_DIR, SCENARIO_FOLDERS, get_scenario_color, get_subject_group


RESULTS_BASE_DIR = os.path.join(PROJECT_ROOT, "Results_Trajectory")

# =========================
# 全局字体设置
# =========================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


# =========================
# 楼层筛选参数
# =========================

HALL_Y_MIN = -12.0
HALL_Y_MAX = -11.6

PLATFORM_Y_MIN = -15.8
PLATFORM_Y_MAX = -15.6

HALL_SCALE = 12.85
PLATFORM_SCALE = 25.5


# =========================
# 单图输出参数
# =========================

OUTPUT_DPI = 300
OUTPUT_WIDTH_PX = 1500
OUTPUT_HEIGHT_PX = 2500

FIG_WIDTH_INCH = OUTPUT_WIDTH_PX / OUTPUT_DPI
FIG_HEIGHT_INCH = OUTPUT_HEIGHT_PX / OUTPUT_DPI

# 单图绘图区位置
AX_POSITION = [0.00, 0.00, 1.00, 0.93]

# 单图图例位置，数值越大越靠上
LEGEND_Y = 0.985


# =========================
# 拼接图输出参数
# =========================

COMBINED_OUTPUT_DPI = 300

# 拼接图总尺寸：宽度缩短
COMBINED_OUTPUT_WIDTH_PX = 2400
COMBINED_OUTPUT_HEIGHT_PX = 2500

COMBINED_FIG_WIDTH_INCH = COMBINED_OUTPUT_WIDTH_PX / COMBINED_OUTPUT_DPI
COMBINED_FIG_HEIGHT_INCH = COMBINED_OUTPUT_HEIGHT_PX / COMBINED_OUTPUT_DPI

# 两图间距进一步收紧：左右子图各向内平移的像素数（面板间距 ≈ 198 → 78 px）
COMBINED_PANEL_PULL_PX = 60

COMBINED_HALL_AX_POSITION = [0.06 + COMBINED_PANEL_PULL_PX / COMBINED_OUTPUT_WIDTH_PX, 0.15, 0.52, 0.72]
COMBINED_PLATFORM_AX_POSITION = [0.59 - COMBINED_PANEL_PULL_PX / COMBINED_OUTPUT_WIDTH_PX, 0.23, 0.28, 0.56]

# 右侧裁剪参数 a：最终拼接图从右向内裁剪的像素数（300 dpi）
COMBINED_RIGHT_CROP_PX = 200

# 场景名标题横坐标：画幅居中 0.5，并随右侧裁剪 a 向左平移 0.5a
COMBINED_TITLE_X = 0.5 - (0.5 * COMBINED_RIGHT_CROP_PX / COMBINED_OUTPUT_DPI) / COMBINED_FIG_WIDTH_INCH

# 拼接图标题位置：va="bottom"，取值为标题文字底边高度
COMBINED_TITLE_Y = 0.8

# Hall / Platform 标题统一放在同一水平高度
COMBINED_LAYER_TITLE_Y = 0.195


# =========================
# 背景图设置
# =========================
# 背景图放置位置：
# PROJECT_ROOT/Background_Trajectory/Hall.png
# PROJECT_ROOT/Background_Trajectory/Platform.png

BACKGROUND_IMAGES = {
    "Hall": os.path.join(PROJECT_ROOT, "Background_Trajectory", "Hall.png"),
    "Platform": os.path.join(PROJECT_ROOT, "Background_Trajectory", "Platform.png"),
}

# 背景图对应的坐标范围
# 格式：(x_min, x_max, y_min, y_max)
#
# 如果设置为 None：
# 程序会根据所有轨迹自动计算范围，并根据背景图宽高比自动扩展范围，
# 避免背景图被拉伸变形。
#
# 如果想让背景图和轨迹更精确对齐，建议手动设置，例如：
# "Hall": (-600, 600, -1000, 1000)
BACKGROUND_EXTENTS = {
    "Hall": None,
    "Platform": None,
}

# 背景图方向
# 一般用 upper，保持图片本身方向
# 如果背景图上下颠倒，再改成 lower
BACKGROUND_ORIGIN = "upper"


# =========================
# 轨迹变换设置
# =========================

# 是否对轨迹进行上下翻转
# True 表示 y = -y
TRAJ_VERTICAL_FLIP = False


# =========================
# 图例设置
# =========================

def make_legend_elements(scenarios, markersize=5):
    return [
        Line2D([0], [0], marker="o", color=get_scenario_color(scenario),
               label=display_scenario_name(scenario), markersize=markersize,
               linestyle="")
        for scenario in scenarios
    ]


# =========================
# 显示名称设置
# =========================

# 图面显示名称统一：Full → Multimodal、ArrowRibbon → Arrow。
# 仅作用于图面显示文本；内部键名、目录名、文件名与路径均保持原样。
SCENARIO_DISPLAY_NAMES = {
    "Full": "Multimodal",
    "ArrowRibbon": "Arrow",
}


def display_scenario_name(scenario_name):
    """返回图面显示用的场景名（只替换类别前缀，例如 Full-LowSmoke → Multimodal-LowSmoke）。"""
    prefix, sep, rest = scenario_name.partition("-")
    return SCENARIO_DISPLAY_NAMES.get(prefix, prefix) + sep + rest


def get_all_subject_files(scenario_path):
    files = []

    for root, dirs, filenames in os.walk(scenario_path):
        for filename in filenames:
            if "Subject_Simulation_Info" in filename and filename.endswith(".csv"):
                files.append(os.path.join(root, filename))

    return sorted(files)


def extract_traj_by_y_range(file_path, y_min, y_max):
    traj = []

    try:
        df = pd.read_csv(file_path, index_col=False)
    except Exception:
        return traj

    if df.empty or df.shape[1] < 4:
        return traj

    mat = df.to_numpy()

    for row in mat:
        try:
            y_val = float(row[2])

            if y_min < y_val < y_max:
                x_val = float(row[1])
                z_val = float(row[3])
                traj.append((x_val, z_val))

        except (ValueError, IndexError):
            continue

    return traj


def transform_traj(traj, scale):
    """
    轨迹坐标变换。

    原始逻辑：
        x_new = -x * scale
        y_new = -z * scale

    如果 TRAJ_VERTICAL_FLIP = True：
        y_new = -y_new
    """

    transformed = []

    for x, z in traj:
        x_new = -x * scale
        y_new = -z * scale

        if TRAJ_VERTICAL_FLIP:
            y_new = -y_new

        transformed.append((x_new, y_new))

    return transformed


def collect_scenario_trajs(scenario_folder):
    scenario_path = os.path.join(RAW_DATA_DIR, scenario_folder)

    if not os.path.exists(scenario_path):
        print(f"[警告] 目录不存在: {scenario_folder}")
        return None

    files = get_all_subject_files(scenario_path)

    if not files:
        print(f"[跳过] 无 Subject_Simulation_Info 文件: {scenario_folder}")
        return None

    hall_group_trajs = {scenario_folder: []}
    platform_group_trajs = {scenario_folder: []}

    for file_path in tqdm(files, desc=scenario_folder, unit="file"):
        group = get_subject_group(file_path)

        if group is None:
            continue

        hall_traj_raw = extract_traj_by_y_range(
            file_path,
            HALL_Y_MIN,
            HALL_Y_MAX
        )
        hall_traj = transform_traj(hall_traj_raw, HALL_SCALE)
        hall_group_trajs[scenario_folder].append(hall_traj)

        platform_traj_raw = extract_traj_by_y_range(
            file_path,
            PLATFORM_Y_MIN,
            PLATFORM_Y_MAX
        )
        platform_traj = transform_traj(platform_traj_raw, PLATFORM_SCALE)
        platform_group_trajs[scenario_folder].append(platform_traj)

    return {
        "Hall": hall_group_trajs,
        "Platform": platform_group_trajs,
    }


def get_extent_from_trajs(group_trajs, margin_ratio=0.04):
    xs = []
    ys = []

    for group in group_trajs:
        for traj in group_trajs[group]:
            for x, y in traj:
                xs.append(x)
                ys.append(y)

    if not xs or not ys:
        return None

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    x_margin = (x_max - x_min) * margin_ratio
    y_margin = (y_max - y_min) * margin_ratio

    if x_margin == 0:
        x_margin = 1.0

    if y_margin == 0:
        y_margin = 1.0

    return (
        x_min - x_margin,
        x_max + x_margin,
        y_min - y_margin,
        y_max + y_margin
    )


def merge_extents(extents):
    valid_extents = [e for e in extents if e is not None]

    if not valid_extents:
        return None

    x_min = min(e[0] for e in valid_extents)
    x_max = max(e[1] for e in valid_extents)
    y_min = min(e[2] for e in valid_extents)
    y_max = max(e[3] for e in valid_extents)

    return x_min, x_max, y_min, y_max


def get_background_aspect(layer):
    """
    获取背景图宽高比：width / height
    """

    bg_path = BACKGROUND_IMAGES.get(layer)

    if not bg_path:
        return None

    if not os.path.exists(bg_path):
        return None

    try:
        img = mpimg.imread(bg_path)
    except Exception:
        return None

    height = img.shape[0]
    width = img.shape[1]

    if height == 0:
        return None

    return width / height


def adjust_extent_to_background_aspect(extent, layer):
    """
    根据背景图宽高比修正 extent，避免背景图被拉伸。

    imshow 会把图片填充到 extent 对应的矩形区域。
    如果 extent 宽高比和图片原始宽高比不同，图片会变形。
    所以这里只扩展 x 或 y 的坐标范围，不压缩图片。
    """

    if extent is None:
        return None

    bg_aspect = get_background_aspect(layer)

    if bg_aspect is None:
        return extent

    x_min, x_max, y_min, y_max = extent

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    x_range = x_max - x_min
    y_range = y_max - y_min

    if x_range <= 0 or y_range <= 0:
        return extent

    current_aspect = x_range / y_range

    if current_aspect < bg_aspect:
        # 当前绘图范围太窄，扩展 x 方向
        new_x_range = y_range * bg_aspect
        x_min = x_center - new_x_range / 2
        x_max = x_center + new_x_range / 2

    elif current_aspect > bg_aspect:
        # 当前绘图范围太宽，扩展 y 方向
        new_y_range = x_range / bg_aspect
        y_min = y_center - new_y_range / 2
        y_max = y_center + new_y_range / 2

    return x_min, x_max, y_min, y_max


def draw_background(ax, layer, extent):
    bg_path = BACKGROUND_IMAGES.get(layer)

    if not bg_path:
        return

    if not os.path.exists(bg_path):
        print(f"  [提示] 未找到 {layer} 背景图: {bg_path}")
        return

    if extent is None:
        print(f"  [提示] {layer} 没有有效 extent，跳过背景叠加")
        return

    try:
        bg_img = mpimg.imread(bg_path)
    except Exception as e:
        print(f"  [警告] 背景图读取失败: {bg_path}, {e}")
        return

    ax.imshow(
        bg_img,
        extent=extent,
        origin=BACKGROUND_ORIGIN,
        zorder=0
    )


def draw_trajs_on_axis(ax, group_trajs, layer, fixed_extent):
    """
    在指定 ax 上绘制某一层轨迹。
    该函数同时供单图和拼接图调用。
    """

    draw_background(ax, layer, fixed_extent)

    for group in group_trajs:
        trajs = [t for t in group_trajs[group] if t]

        if not trajs:
            continue

        color = get_scenario_color(group)

        for traj in trajs:
            x_coords = [p[0] for p in traj]
            y_coords = [p[1] for p in traj]

            ax.plot(
                x_coords,
                y_coords,
                marker="o",
                color=color,
                markersize=0.06,
                linewidth=0.02,
                zorder=2
            )

    ax.set_aspect("equal", adjustable="box")

    if fixed_extent is not None:
        ax.set_xlim(fixed_extent[0], fixed_extent[1])
        ax.set_ylim(fixed_extent[2], fixed_extent[3])

    ax.axis("off")


def draw_scenario_traj(group_trajs, scenario_name, layer, out_dir, fixed_extent):
    """
    绘制单个场景的单层轨迹：
    1. 背景图保持原始宽高比，不拉伸；
    2. 轨迹叠加在背景图上；
    3. 输出图片尺寸固定为 1500 x 2500 px；
    4. 图例放在顶部。
    """

    plt.rcParams["figure.dpi"] = OUTPUT_DPI
    plt.rcParams["font.family"] = "Times New Roman"

    fig = plt.figure(
        figsize=(FIG_WIDTH_INCH, FIG_HEIGHT_INCH),
        dpi=OUTPUT_DPI
    )

    position = AX_POSITION if len(group_trajs) <= 2 else [0.00, 0.00, 1.00, 0.85]
    ax = fig.add_axes(position)

    draw_trajs_on_axis(
        ax=ax,
        group_trajs=group_trajs,
        layer=layer,
        fixed_extent=fixed_extent
    )

    legend_elements = make_legend_elements(group_trajs, markersize=5)

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, LEGEND_Y),
        ncol=2 if len(group_trajs) > 1 else 1,
        fontsize=10,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.2
    )

    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{scenario_name}_{layer}.png")

    fig.savefig(
        out_path,
        dpi=OUTPUT_DPI,
        transparent=False,
        facecolor="white"
    )

    plt.close(fig)

    print(f"  已保存单图: {out_path}")


def draw_scenario_combined_traj(result, scenario_name, out_dir, fixed_extents):
    """
    将同一个场景的 Hall 和 Platform 拼接成一张图。

    当前版本：
    1. Hall 较大；Platform 缩小一些，两图间距进一步收紧（见 COMBINED_PANEL_PULL_PX）；
    2. Hall / Platform 标题放在各自图像下方，不标注 a/b 编号；
    3. 标题为纯文字，无文本框；
    4. 顶部显示场景名；合并图增加实际条件的统一配色图例；
    5. 输出按内容紧裁，并从右侧向内裁剪 a（COMBINED_RIGHT_CROP_PX），
       场景名标题相应向左平移 0.5a 保持画幅居中。
    """

    plt.rcParams["figure.dpi"] = COMBINED_OUTPUT_DPI
    plt.rcParams["font.family"] = "Times New Roman"

    fig = plt.figure(
        figsize=(COMBINED_FIG_WIDTH_INCH, COMBINED_FIG_HEIGHT_INCH),
        dpi=COMBINED_OUTPUT_DPI
    )

    fig.suptitle(
        display_scenario_name(scenario_name),
        fontsize=22,
        x=COMBINED_TITLE_X,
        fontweight="bold",
        y=COMBINED_TITLE_Y,
        va="bottom",
        family="Times New Roman"
    )

    scenarios = list(result["Hall"])
    if len(scenarios) > 1:
        fig.legend(
            handles=make_legend_elements(scenarios), loc="lower center",
            bbox_to_anchor=(COMBINED_TITLE_X, 0.84), ncol=2,
            fontsize=10, frameon=False, handletextpad=0.4,
            columnspacing=1.2,
        )

    ax_hall = fig.add_axes(COMBINED_HALL_AX_POSITION)
    ax_platform = fig.add_axes(COMBINED_PLATFORM_AX_POSITION)

    draw_trajs_on_axis(
        ax=ax_hall,
        group_trajs=result["Hall"],
        layer="Hall",
        fixed_extent=fixed_extents["Hall"]
    )

    draw_trajs_on_axis(
        ax=ax_platform,
        group_trajs=result["Platform"],
        layer="Platform",
        fixed_extent=fixed_extents["Platform"]
    )

    # Hall / Platform 标题放在同一水平高度，居中，无文本框
    hall_title_x = COMBINED_HALL_AX_POSITION[0] + COMBINED_HALL_AX_POSITION[2] / 2
    platform_title_x = COMBINED_PLATFORM_AX_POSITION[0] + COMBINED_PLATFORM_AX_POSITION[2] / 2

    hall_title_y = COMBINED_LAYER_TITLE_Y
    platform_title_y = COMBINED_LAYER_TITLE_Y

    fig.text(
        hall_title_x,
        hall_title_y,
        "Hall",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
        family="Times New Roman"
    )

    fig.text(
        platform_title_x,
        platform_title_y,
        "Platform",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
        family="Times New Roman"
    )

    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(
        out_dir,
        f"{scenario_name}_Hall_Platform.png"
    )

    # 输出紧裁四周留白（保留 0.02 in 边距），并从右侧向内裁剪 a（COMBINED_RIGHT_CROP_PX）
    # 注意：bbox_inches 传入 Bbox 时 pad_inches 不生效，边距在此显式加入
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    # 汇总图图例可能延伸至右侧，不应用单场景的额外裁剪。
    crop_right_in = COMBINED_RIGHT_CROP_PX / COMBINED_OUTPUT_DPI if len(scenarios) == 1 else 0
    pad_in = 0.02

    save_bbox = Bbox.from_extents(
        tight.x0 - pad_in,
        tight.y0 - pad_in,
        tight.x1 + pad_in - crop_right_in,
        tight.y1 + pad_in
    )

    fig.savefig(
        out_path,
        dpi=COMBINED_OUTPUT_DPI,
        transparent=False,
        facecolor="white",
        bbox_inches=save_bbox
    )

    plt.close(fig)

    print(f"  已保存拼接图: {out_path}")


def count_valid_trajs(group_trajs):
    return sum(
        1
        for group in group_trajs
        for traj in group_trajs[group]
        if traj
    )


def main():
    print(f"\n{'=' * 70}")
    print("轨迹绘制 - 站厅层(Hall) & 站台层(Platform)")
    print(f"{'=' * 70}")
    print(f"工程根目录: {PROJECT_ROOT}")
    print(f"结果输出目录: {RESULTS_BASE_DIR}")
    print(f"场景数量: {len(SCENARIO_FOLDERS)}")
    print(f"单图输出尺寸: {OUTPUT_WIDTH_PX} × {OUTPUT_HEIGHT_PX} px")
    print(f"拼接图输出尺寸: {COMBINED_OUTPUT_WIDTH_PX} × {COMBINED_OUTPUT_HEIGHT_PX} px")
    print(f"{'=' * 70}\n")

    os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

    # 第一步：读取所有场景轨迹
    all_results = {}

    for idx, scenario_folder in enumerate(SCENARIO_FOLDERS, 1):
        print(f"[{idx}/{len(SCENARIO_FOLDERS)}] 读取场景: {scenario_folder}")

        result = collect_scenario_trajs(scenario_folder)

        if result is not None:
            all_results[scenario_folder] = result

    if not all_results:
        print("[结束] 没有可绘制的轨迹数据")
        return

    # 第二步：计算 Hall / Platform 的统一坐标范围
    auto_extents = {
        "Hall": [],
        "Platform": [],
    }

    for scenario_folder, result in all_results.items():
        for layer in ("Hall", "Platform"):
            extent = get_extent_from_trajs(result[layer])
            auto_extents[layer].append(extent)

    fixed_extents = {}

    for layer in ("Hall", "Platform"):
        if BACKGROUND_EXTENTS[layer] is not None:
            raw_extent = BACKGROUND_EXTENTS[layer]
        else:
            raw_extent = merge_extents(auto_extents[layer])

        fixed_extents[layer] = adjust_extent_to_background_aspect(
            raw_extent,
            layer
        )

    print("\n统一绘图范围:")
    print(f"  Hall: {fixed_extents['Hall']}")
    print(f"  Platform: {fixed_extents['Platform']}")

    # 合并时保留各 trial 独立的轨迹及条件颜色，不拼接不同 trial 的点。
    plot_results = dict(all_results)
    guidance_order = ("StaticSign", "ArrowRibbon", "MiniMap", "TextBar", "Full")
    for guidance in guidance_order:
        members = [f"{guidance}-{smoke}" for smoke in ("LowSmoke", "HighSmoke")]
        if not all(name in all_results for name in members):
            raise RuntimeError(f"Incomplete guidance conditions: {guidance}")
        plot_results[f"{guidance}-AllSmoke"] = {
            layer: {name: all_results[name][layer][name] for name in members}
            for layer in ("Hall", "Platform")
        }
    members = [f"{guidance}-{smoke}" for guidance in guidance_order
               for smoke in ("LowSmoke", "HighSmoke")]
    plot_results["All-10-Conditions"] = {
        layer: {name: all_results[name][layer][name] for name in members}
        for layer in ("Hall", "Platform")
    }

    # 记录各图的源条件、颜色、有效轨迹数，方便核对合并结果。
    audit = {
        name: {
            "scenarios": list(result["Hall"]),
            "colors": {key: get_scenario_color(key) for key in result["Hall"]},
            "valid_trajectories": {
                layer: {key: sum(bool(t) for t in trajectories)
                        for key, trajectories in result[layer].items()}
                for layer in ("Hall", "Platform")
            },
        }
        for name, result in plot_results.items()
    }

    # 第三步：正式绘图（10 个单条件 + 5 个模态合并 + 1 个全条件合并）
    for idx, scenario_folder in enumerate(tqdm(plot_results, desc="Rendering", unit="set"), 1):
        print(f"\n[{idx}/{len(plot_results)}] 绘制场景: {scenario_folder}")

        result = plot_results[scenario_folder]
        out_dir = os.path.join(RESULTS_BASE_DIR, scenario_folder)

        hall_total = count_valid_trajs(result["Hall"])
        platform_total = count_valid_trajs(result["Platform"])

        print(f"  Hall 有效轨迹: {hall_total} 条")
        print(f"  Platform 有效轨迹: {platform_total} 条")

        # 原功能：分别绘制 Hall 和 Platform 单图
        draw_scenario_traj(
            result["Hall"],
            scenario_folder,
            "Hall",
            out_dir,
            fixed_extents["Hall"]
        )

        draw_scenario_traj(
            result["Platform"],
            scenario_folder,
            "Platform",
            out_dir,
            fixed_extents["Platform"]
        )

        # 新增功能：绘制 Hall + Platform 拼接图
        draw_scenario_combined_traj(
            result=result,
            scenario_name=scenario_folder,
            out_dir=out_dir,
            fixed_extents=fixed_extents
        )

    print(f"\n{'=' * 70}")
    with open(os.path.join(RESULTS_BASE_DIR, "trajectory_plot_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"fixed_extents": fixed_extents, "plots": audit}, handle, indent=2)
    print("全部场景处理完毕")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()