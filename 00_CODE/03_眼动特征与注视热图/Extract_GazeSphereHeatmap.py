import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from scipy.stats import gaussian_kde
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from common import PROJECT_ROOT, GAZE_TOPO_DIR, SCENARIO_FOLDERS

# 输入目录（复用 RelativePosition 的源数据文件）
RELATIVE_POS_BASE_DIR = os.path.join(PROJECT_ROOT, "Results_RelativePosition")
# 输出目录
RESULTS_BASE_DIR = os.path.join(GAZE_TOPO_DIR, "Results_GazeSphereHeatmap")

HEMISPHERE_RADIUS = 0.05

# 眼位距脚底（SubjectWorld 位置，rig root/地面高度）的恒定高度 (单位: m)。
# 实验 rig 眼高固定不变，视线真实起点 = Subject 位置 + 该眼高。
EYE_HEIGHT_M = 1.771

def get_custom_cmap():
    """
    构造自定义的颜色映射：
    - 密度趋近 0 时，显示半透明的淡灰色 [0.8, 0.8, 0.8, 0.3]
    - 密度大于阈值时，平滑过渡到正常的热力图颜色 (jet)，同时带有一定透明度
    """
    jet = cm.get_cmap('jet')
    colors = jet(np.linspace(0, 1, 256))
    base_color = np.array([0.8, 0.8, 0.8, 0.3])
    
    threshold_idx = 25  # 前10%左右作为过渡区
    for i in range(256):
        if i < threshold_idx:
            alpha = i / float(threshold_idx)
            jet_color = colors[i].copy()
            jet_color[3] = 0.8  # 热力图部分保留一定透明度以看清形状
            # 颜色线性插值混合
            colors[i] = base_color * (1 - alpha) + jet_color * alpha
        else:
            colors[i][3] = 0.8
            
    return LinearSegmentedColormap.from_list('custom_hemisphere', colors)

def calculate_3d_relative_vectors(df):
    """
    基于世界坐标计算相对三维向量
    """
    sub_x = df['SubjectWorldX'].to_numpy(dtype=float)
    sub_y = df['SubjectWorldY'].to_numpy(dtype=float)
    sub_z = df['SubjectWorldZ'].to_numpy(dtype=float)
    sub_rot_y = df['SubjectRotY_deg'].to_numpy(dtype=float)
    
    gaze_x = df['GazeWorldX'].to_numpy(dtype=float)
    gaze_y = df['GazeWorldY'].to_numpy(dtype=float)
    gaze_z = df['GazeWorldZ'].to_numpy(dtype=float)
    
    # 平移
    dx = gaze_x - sub_x
    dy = gaze_y - sub_y
    dz = gaze_z - sub_z
    
    # 绕Y轴旋转 (复用2D逻辑的向量化版本)
    rotation_rad = np.radians(sub_rot_y)
    cos_rad = np.cos(rotation_rad)
    sin_rad = np.sin(rotation_rad)
    
    local_x = cos_rad * dx - sin_rad * dz
    local_z = sin_rad * dx + cos_rad * dz
    local_y = dy
    
    # 返回相对三维坐标：(PlotX, PlotY, PlotZ)
    # 注意：画图时我们需要 Y 作为前方（Z > 0 对应的方向）
    # 因此映射关系为: PlotX = local_x (右), PlotY = local_z (前), PlotZ = local_y (上)
    return np.column_stack([local_x, local_z, local_y])

def draw_sphere_heatmap(points_3d, scenario_name, output_dir, pbar=None):
    """
    绘制半球面热力图
    """
    if pbar: pbar.set_postfix_str("Data Norm")
    
    # 1. 过滤与归一化
    norms = np.linalg.norm(points_3d, axis=1)
    # 仅保留前方的数据 (PlotY > 0)
    valid_mask = (norms > 1e-5) & (points_3d[:, 1] > 0)
    
    if np.sum(valid_mask) < 10:
        if pbar: pbar.set_postfix_str("Skipped: Not enough valid data")
        if pbar: pbar.update(5)
        return np.nan
        
    valid_points = points_3d[valid_mask]
    # 视线真实起点在眼位：SubjectWorld 位置是脚底/rig root（地面高度），
    # 眼位在其上方恒定 EYE_HEIGHT_M。方向向量须以眼位为原点，
    # 故竖直分量（第 3 列，向上）先减去眼高，再归一化得到真实注视方向。
    # （与 Extract_GazePlaneHeatmap 同源：那里在平面投影时修正，这里在方向归一化前修正）
    valid_points[:, 2] -= EYE_HEIGHT_M
    unit_vectors = valid_points / np.linalg.norm(valid_points, axis=1)[:, np.newaxis]
    p_group = unit_vectors * HEMISPHERE_RADIUS
    
    if pbar: pbar.update(1)
    if pbar: pbar.set_postfix_str("Meshgrid")
    
    # 2. 构造半球面 Meshgrid
    # u: X-Y 平面上的角度 (0 到 pi，以保证 Y >= 0)
    # v: 垂直于 Z 轴的角度 (0 到 pi，Z 从顶到底)
    u = np.linspace(0, np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    U, V = np.meshgrid(u, v)
    
    X_mesh = HEMISPHERE_RADIUS * np.sin(V) * np.cos(U)
    Y_mesh = HEMISPHERE_RADIUS * np.sin(V) * np.sin(U)
    Z_mesh = HEMISPHERE_RADIUS * np.cos(V)
    
    if pbar: pbar.update(1)
    if pbar: pbar.set_postfix_str("KDE")
    
    # 3. 计算 KDE 密度
    try:
        kde = gaussian_kde(p_group.T, bw_method='scott')
    except Exception as e:
        if pbar: pbar.set_postfix_str("Skipped: KDE failed")
        if pbar: pbar.update(3)
        return np.nan
        
    mesh_points = np.vstack([X_mesh.ravel(), Y_mesh.ravel(), Z_mesh.ravel()])
    density = kde(mesh_points)
    density = density.reshape(X_mesh.shape)
    
    # 密度归一化至 [0, 1]
    d_min, d_max = density.min(), density.max()
    if d_max - d_min > 1e-8:
        density_norm = (density - d_min) / (d_max - d_min)
    else:
        density_norm = np.zeros_like(density)

    # ── 计算半球面上的香农熵 ──
    # 球面面积元 dA = R² sin(V) du dv → 权重 w = sin(V)
    weights = np.sin(V)
    prob = density * weights
    prob_sum = prob.sum()
    if prob_sum > 1e-12:
        prob /= prob_sum
        entropy = -np.sum(prob * np.log2(prob + 1e-30))
    else:
        entropy = np.nan

    if pbar: pbar.update(1)
    if pbar: pbar.set_postfix_str("Plotting PNG")
    
    # 4. 颜色映射
    custom_cmap = get_custom_cmap()
    facecolors = custom_cmap(density_norm)
    
    # 5. 绘图 - 双子图布局
    # 为了多线程安全，使用 Figure 构造函数而非 pyplot.figure()
    fig = matplotlib.figure.Figure(figsize=(16, 7), dpi=300)
    
    # 整个图的标题（含香农熵）
    entropy_str = f"{entropy:.4f}" if np.isfinite(entropy) else "N/A"
    fig.suptitle(f"Directional Distribution of Gaze Vectors\nGroup: {scenario_name}    H = {entropy_str} bit",
                 fontsize=20, y=0.95)
    
    def setup_axis(ax, elev, azim, hide_y=False):
        ax.plot_surface(X_mesh, Y_mesh, Z_mesh, facecolors=facecolors, rstride=1, cstride=1, antialiased=True, shade=False)
        ax.set_xlabel('X')
        
        if not hide_y:
            ax.set_ylabel('Y')
        else:
            # 隐藏Y轴的文字标注和刻度标签，避免正面看时重叠
            ax.set_ylabel('')
            ax.set_yticks([])
            
        ax.set_zlabel('Z')
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim([-HEMISPHERE_RADIUS, HEMISPHERE_RADIUS])
        ax.set_ylim([0, HEMISPHERE_RADIUS])
        ax.set_zlim([-HEMISPHERE_RADIUS, HEMISPHERE_RADIUS])
        ax.set_box_aspect([1, 0.5, 1])

    # 左子图：侧前方等轴测视角
    ax1 = fig.add_subplot(121, projection='3d')
    setup_axis(ax1, elev=30, azim=45, hide_y=False)
    
    # 右子图：正前方视角（向半球凸起看去）
    ax2 = fig.add_subplot(122, projection='3d')
    setup_axis(ax2, elev=0, azim=90, hide_y=True)
    
    # 共用的 Colorbar
    m = cm.ScalarMappable(cmap=custom_cmap)
    m.set_array(density)
    
    # 显式指定 colorbar 的位置，使得整个布局紧凑且不影响子图大小
    # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(m, cax=cbar_ax)
    cbar.set_label('Density', fontsize=14)
    
    # 调整布局留白，避免显性裁切导致输出尺寸不一致
    fig.subplots_adjust(left=0.05, right=0.85, bottom=0.1, top=0.85, wspace=0.15)
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{scenario_name}_SphereHeatmap.png")
    
    # 多线程环境下必须绕过 plt 全局状态机进行保存
    # 显式附加一个 Agg Canvas 到底层，并直接使用 Figure 自带的 savefig 方法
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig)
    fig.savefig(out_path, dpi=300, transparent=True)
    # 清理 Figure 释放内存
    fig.clear()
    
    if pbar: pbar.update(1)
    if pbar: pbar.set_postfix_str("Plotting HTML")
    
    # 6. 生成交互式 HTML (使用 Plotly)
    if go is not None:
        save_interactive_html(X_mesh, Y_mesh, Z_mesh, density, p_group, scenario_name, output_dir)
    else:
        if pbar: pbar.set_postfix_str("No plotly, skip HTML")
        
    if pbar: pbar.update(1)
    if pbar: pbar.set_postfix_str("Done")
    return entropy


def save_interactive_html(X_mesh, Y_mesh, Z_mesh, density, p_group, scenario_name, output_dir):
    """
    使用 Plotly 构造可交互的 3D HTML 文件，并将数据完全嵌入（单体文件）
    """
    # 构造与 Matplotlib 一致的 colorscale
    jet = cm.get_cmap('jet')
    base_color = np.array([0.8, 0.8, 0.8, 0.3])
    threshold_idx = 25
    
    plotly_colorscale = []
    for i in range(256):
        frac = i / 255.0
        if i < threshold_idx:
            alpha = i / float(threshold_idx)
            jet_color = np.array(jet(i))
            jet_color[3] = 0.8
            c = base_color * (1 - alpha) + jet_color * alpha
        else:
            c = np.array(jet(i))
            c[3] = 0.8
        # 转为 rgba 字符串
        r, g, b = int(c[0]*255), int(c[1]*255), int(c[2]*255)
        plotly_colorscale.append([frac, f'rgba({r},{g},{b},{c[3]:.3f})'])
        
    fig = go.Figure()
    
    # 1. 添加曲面热力图
    fig.add_trace(go.Surface(
        x=X_mesh, y=Y_mesh, z=Z_mesh,
        surfacecolor=density,
        colorscale=plotly_colorscale,
        cmin=density.min(),
        cmax=density.max(),
        colorbar=dict(title='Density', x=0.85),
        showscale=True,
        name='Density Surface'
    ))
    
    # 2. 添加散点图
    # 稍微给散点加一点向外的偏移（或者直接用原坐标），避免与曲面发生Z-fighting
    # 这里直接用原 p_group，因为 Plotly 散点是渲染在最顶层的
    fig.add_trace(go.Scatter3d(
        x=p_group[:, 0],
        y=p_group[:, 1],
        z=p_group[:, 2],
        mode='markers',
        marker=dict(
            size=1.5,
            color='gray',
            opacity=0.1
        ),
        name='Gaze Points',
        hoverinfo='none'  # 散点太多，关闭悬停提示避免卡顿
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Directional Distribution of Gaze Vectors<br>Group: {scenario_name}",
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top'
        ),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(range=[-HEMISPHERE_RADIUS, HEMISPHERE_RADIUS]),
            yaxis=dict(range=[0, HEMISPHERE_RADIUS]),
            zaxis=dict(range=[-HEMISPHERE_RADIUS, HEMISPHERE_RADIUS]),
            aspectratio=dict(x=1, y=0.5, z=1),
            camera=dict(
                eye=dict(x=1.25, y=1.25, z=0.8) # 初始视角: 侧前方等轴测
            )
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    html_path = os.path.join(output_dir, f"{scenario_name}_SphereHeatmap.html")
    # include_plotlyjs=True 会把整个库源码明文打进去，生成一个完全脱机的巨大独立 html
    fig.write_html(html_path, include_plotlyjs=True)
    # print(f"  交互式HTML已保存：{html_path}")


def process_scenario(scenario_folder, position=0):
    """处理单个场景目录的数据，提取向量并绘制双子图与HTML"""
    # 如果没有 tqdm，就简单打印；如果有 tqdm，就创建进度条
    if tqdm is None:
        print(f"  [开始] 组别: {scenario_folder}")
        pbar = None
    else:
        # 总共 5 个大步骤：Data Norm -> Meshgrid -> KDE -> PNG -> HTML
        pbar = tqdm(total=5, desc=f"{scenario_folder[:20]:<20}", position=position, leave=True)
        
    input_csv = os.path.join(RELATIVE_POS_BASE_DIR, scenario_folder, "AllSubjects_RelativePosInfo.csv")
    
    if not os.path.exists(input_csv):
        msg = f"  [跳过] {scenario_folder} 找不到数据文件"
        if pbar: pbar.set_postfix_str("Skipped: No data file"); pbar.update(5); pbar.close()
        return {"场景": scenario_folder, "香农熵(bit)": np.nan}

    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        msg = f"  [错误] {scenario_folder} 无法读取 {input_csv}"
        if pbar: pbar.set_postfix_str("Error: Read failed"); pbar.update(5); pbar.close()
        return {"场景": scenario_folder, "香农熵(bit)": np.nan}

    # 确保必需的列存在
    required_cols = ['SubjectWorldX', 'SubjectWorldY', 'SubjectWorldZ', 'SubjectRotY_deg',
                     'GazeWorldX', 'GazeWorldY', 'GazeWorldZ']
    if not all(c in df.columns for c in required_cols):
        msg = f"  [跳过] {scenario_folder} 数据列不完整"
        if pbar: pbar.set_postfix_str("Skipped: Missing cols"); pbar.update(5); pbar.close()
        return {"场景": scenario_folder, "香农熵(bit)": np.nan}

    # 剔除无效行
    df = df.dropna(subset=required_cols)
    if len(df) == 0:
        msg = f"  [跳过] {scenario_folder} 剔除空值后无有效数据"
        if pbar: pbar.set_postfix_str("Skipped: Empty data"); pbar.update(5); pbar.close()
        return {"场景": scenario_folder, "香农熵(bit)": np.nan}
        
    if pbar: pbar.set_postfix_str("Calc Vectors")
    points_3d = calculate_3d_relative_vectors(df)
    
    scenario_output_dir = os.path.join(RESULTS_BASE_DIR, scenario_folder)
    entropy = draw_sphere_heatmap(points_3d, scenario_folder, scenario_output_dir, pbar=pbar)
    
    if pbar: pbar.close()
    return {"场景": scenario_folder, "香农熵(bit)": entropy}


def main():
    print(f"\n{'=' * 70}")
    print(f"3D眼动相对位置 - 半球面热力图生成 (多线程并行版 + tqdm)")
    print(f"{'=' * 70}")
    print(f"眼高(恒定, 竖直分量已减 {EYE_HEIGHT_M} m, 视线起点取眼位)")
    print(f"结果输出目录: {RESULTS_BASE_DIR}")
    print(f"处理组别(场景)数量: {len(SCENARIO_FOLDERS)}")
    print(f"{'=' * 70}\n")
    
    os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
    
    # 启用10个线程进行并行处理
    entropy_results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_scenario = {}
        for idx, scenario_folder in enumerate(SCENARIO_FOLDERS):
            future = executor.submit(process_scenario, scenario_folder, idx)
            future_to_scenario[future] = scenario_folder
            
        for future in as_completed(future_to_scenario):
            try:
                result = future.result()
                if isinstance(result, dict):
                    entropy_results.append(result)
            except Exception as exc:
                scenario_folder = future_to_scenario[future]
                entropy_results.append({"场景": scenario_folder, "香农熵(bit)": np.nan})
                if tqdm is not None:
                    tqdm.write(f"  [异常] 组别 {scenario_folder} 产生错误: {exc}")
                else:
                    print(f"  [异常] 组别 {scenario_folder} 产生错误: {exc}")

    # 输出香农熵汇总 CSV
    if entropy_results:
        df_entropy = pd.DataFrame(entropy_results).sort_values("场景")
        entropy_csv = os.path.join(RESULTS_BASE_DIR, "ShannonEntropy_Summary.csv")
        df_entropy.to_csv(entropy_csv, index=False, encoding="utf-8-sig")
        print(f"\n香农熵汇总已保存: {entropy_csv}")
        print(df_entropy.to_string(index=False))

    print(f"\n{'=' * 70}")
    print(f"全部处理完成！")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()
