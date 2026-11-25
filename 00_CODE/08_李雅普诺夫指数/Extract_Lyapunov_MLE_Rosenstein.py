import os
import logging
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from common import PROJECT_ROOT, INDICATORS_DIR, SCENARIO_FOLDERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 数据入口：_3D 脚本输出的 13 列 RelativePosInfo.csv
RELATIVE_POS_BASE_DIR = os.path.join(PROJECT_ROOT, "Results_RelativePosition")
# 结果输出目录
OUTPUT_DIR = os.path.join(INDICATORS_DIR, "Results_LyapunovMLE")

# DFT去噪默认关闭（保留原始速度序列）
ENABLE_DFT_DENOISE = False


# ═══════════════════════════════════════════════════════════════
# 相空间重构
# ═══════════════════════════════════════════════════════════════

def time_delay_embedding(series, tau, m):
    """
    Takens 延迟坐标法相空间重构。
    将一维时间序列重构为 (N - (m - 1) * tau) x m 的高维相空间矩阵。
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    n_out = n - (m - 1) * tau
    if n_out <= 0:
        return np.empty((0, m), dtype=float)
    embedded = np.empty((n_out, m), dtype=float)
    for i in range(m):
        embedded[:, i] = series[i * tau: i * tau + n_out]
    return embedded


# ═══════════════════════════════════════════════════════════════
# ① 互信息法求 τ — 对标论文 Section 2.1
# ═══════════════════════════════════════════════════════════════

def mutual_information(series, max_lag=50, n_bins=None):
    """
    互信息法求最优时间延迟 τ。

    采用直方图估计联合概率与边缘概率，计算 I(τ)。
    返回 I(τ) 的第一个局部极小值对应的 lag 作为 τ。
    若未找到局部极小值，fallback 到全局最小值。

    Parameters
    ----------
    series : 1D array
        输入的一维时间序列（速度序列）。
    max_lag : int
        最大搜索 lag。
    n_bins : int or None
        直方图 bin 数。None 时自动用 Freedman–Diaconis 估计。

    Returns
    -------
    tau : int
        最优时间延迟。
    """
    series = np.asarray(series, dtype=float)
    n = len(series)

    # 自动确定 bin 数
    if n_bins is None:
        iqr = np.percentile(series, 75) - np.percentile(series, 25)
        if iqr <= 0:
            n_bins = 8
        else:
            bin_width = 2.0 * iqr / (n ** (1.0 / 3.0))  # Freedman–Diaconis
            data_range = series.max() - series.min()
            n_bins = max(4, min(32, int(data_range / bin_width))) if data_range > 0 else 8

    s_min, s_max = series.min(), series.max()
    if s_max - s_min < 1e-12:
        return 1
    normed = (series - s_min) / (s_max - s_min)

    mi_values = []
    search_lags = range(1, min(max_lag, n // 4))
    for lag in search_lags:
        x = normed[:n - lag]
        y = normed[lag:]

        H, _, _ = np.histogram2d(x, y, bins=n_bins, range=[[0, 1], [0, 1]])
        P = H / H.sum()
        Px = P.sum(axis=1)
        Py = P.sum(axis=0)

        mi = 0.0
        for i in range(n_bins):
            if Px[i] <= 0:
                continue
            for j in range(n_bins):
                if P[i, j] > 0 and Py[j] > 0:
                    mi += P[i, j] * np.log2(P[i, j] / (Px[i] * Py[j]))
        mi_values.append(max(mi, 0.0))

    mi_values = np.array(mi_values)
    if len(mi_values) < 3:
        return 1

    # 找第一个局部极小值
    for idx in range(1, len(mi_values) - 1):
        if mi_values[idx] < mi_values[idx - 1] and mi_values[idx] <= mi_values[idx + 1]:
            return list(search_lags)[idx]

    # fallback → 全局最小值
    return list(search_lags)[int(np.argmin(mi_values))]


# ═══════════════════════════════════════════════════════════════
# ② FNN 法求 m — 对标论文 Section 2.2
# ═══════════════════════════════════════════════════════════════

def false_nearest_neighbors(series, tau, max_dim=15, threshold=10.0, tolerance=0.02, chunk_size=2000):
    """
    False Nearest Neighbors 法求最小嵌入维数 m。

    从 m=1 开始逐步增大嵌入维数，对每个候选 m：
      - 在 m 维重构空间中找每个点的最近邻
      - 若在 m+1 维空间中距离比 Rd = |x_{i+mτ} - x_{j+mτ}| / R_m > threshold
        则判定为虚假近邻
      - 当 FNN 比例 < tolerance 时停止，返回当前 m。

    实现细节：保留原始 cdist 计算逻辑，但通过分块（chunk）调用 cdist
    避免单次申请完整 NxN 距离矩阵（>1 GB 时易在 worker 进程中触发 OOM）。
    每次仅临时持有 chunk_size x N 的小块，内存峰值可控。

    Parameters
    ----------
    series : 1D array
    tau : int
        时间延迟。
    max_dim : int
        最大搜索维数。
    threshold : float
        判定虚假近邻的 Rd 阈值（典型值 10~15，论文中用 R_thr）。
    tolerance : float
        FNN 比例阈值，低于此值即认为展开充分。
    chunk_size : int
        分块大小，控制单次 cdist 的内存占用。默认 2000。

    Returns
    -------
    m : int
        最小嵌入维数。若 max_dim 内未满足 tolerance 则返回 max_dim。
    """
    series = np.asarray(series, dtype=float)
    n = len(series)

    for m in range(1, max_dim + 1):
        n_embed = n - m * tau
        if n_embed < 20:
            return max(1, m - 1)

        embedded_m = time_delay_embedding(series, tau, m)
        embedded_m1 = time_delay_embedding(series, tau, m + 1)
        n_pts = min(embedded_m.shape[0], embedded_m1.shape[0]) - 1
        if n_pts < 10:
            return max(1, m)

        # 分块 cdist，避免一次性分配 NxN 全量距离矩阵
        nearest = np.zeros(n_pts, dtype=int)
        R_m = np.zeros(n_pts)
        for i_start in range(0, n_pts, chunk_size):
            i_end = min(i_start + chunk_size, n_pts)
            block_len = i_end - i_start
            # cdist 计算当前块到所有点的距离 (block_len x n_pts)
            dist_block = cdist(embedded_m[i_start:i_end], embedded_m[:n_pts])
            # 排除自身：将当前块对角位置设为 inf
            row_idx = np.arange(block_len)
            dist_block[row_idx, i_start + row_idx] = np.inf
            block_argmin = np.argmin(dist_block, axis=1)
            nearest[i_start:i_end] = block_argmin
            R_m[i_start:i_end] = dist_block[row_idx, block_argmin]

        # 仅在需要时算 m+1 维距离差
        diff_m1 = np.abs(
            embedded_m1[:n_pts, m] - embedded_m1[nearest, m]
        )

        valid = R_m > 1e-12
        if not np.any(valid):
            continue

        fnn_ratio = np.sum(diff_m1[valid] / R_m[valid] > threshold) / valid.sum()
        if fnn_ratio < tolerance:
            return m

    return max_dim


# ═══════════════════════════════════════════════════════════════
# ③ DFT 去噪 — 对标论文 Section 3.2 "Noise reduction"
# ═══════════════════════════════════════════════════════════════

def dft_denoise(series, cutoff_ratio=0.05):
    """
    DFT 阈值去噪。

    对信号做 FFT，将幅值低于 cutoff_ratio * max_amplitude 的频率成分
    置零，再 IFFT 重建。对标论文中利用 DFT 消除线性和非线性噪声的做法。

    Parameters
    ----------
    series : 1D array
    cutoff_ratio : float
        幅值截断比，默认 0.05（保留幅值 ≥ 最大值 5% 的频率）。

    Returns
    -------
    denoised : 1D array，与输入等长。
    """
    series = np.asarray(series, dtype=float)
    n = len(series)

    fft = np.fft.rfft(series)
    amplitudes = np.abs(fft)
    threshold = cutoff_ratio * amplitudes.max()
    fft[amplitudes < threshold] = 0.0

    denoised = np.fft.irfft(fft, n=n)
    return denoised


# ═══════════════════════════════════════════════════════════════
# ⑤ 自适应线性区识别
# ═══════════════════════════════════════════════════════════════

def find_linear_region(y_avg, min_points=5):
    """
    自适应识别 <ln(d)> vs k 曲线的线性缩放区。

    策略：计算局部斜率序列，检测斜率首次出现显著偏离的位置。
    显著偏离定义为：当前斜率偏离初始斜率的幅度超过 3 × 前半段斜率标准差。

    Parameters
    ----------
    y_avg : 1D array
        各 k 步的平均 ln(distance)。
    min_points : int
        线性区至少需要的最小点数。

    Returns
    -------
    end_idx : int
        线性区的最后一个 k 索引（含）。回归时取 [0:end_idx+1]。
    """
    n = len(y_avg)
    if n < min_points + 5:
        return n - 1

    slopes = np.diff(y_avg)
    baseline_std = np.std(slopes[:len(slopes) // 2])
    if baseline_std < 1e-12:
        return n - 1

    for k in range(min_points, len(slopes)):
        if abs(slopes[k] - slopes[0]) > 3.0 * baseline_std:
            return max(min_points, k)

    return n - 1


# ═══════════════════════════════════════════════════════════════
# 核心: Rosenstein MLE — 对标 Information Sciences (2017)
# ═══════════════════════════════════════════════════════════════

def get_lyapunov_rosenstein(points, tau=None, m=None, dt=None,
                            p_period=10, subject_label="", show_progress=True):
    """
    对标 Information Sciences (2017) Harezlak 的完整计算链路：

      ┌─────────────────────────────────────────────────────┐
      │ Nx3 位置 → 合速度(1D) → DFT去噪 → MI求τ → FNN求m   │
      │ → Takens嵌入(m维相空间) → Theiler窗口近邻搜索        │
      │ → 散度追踪 → 全局平均 <ln(d)> → 自适应线性区回归     │
      │ → 除以Δt → λ (nat/s)                                │
      └─────────────────────────────────────────────────────┘

    Parameters
    ----------
    points : (N, 3) array
        3D 相对位置序列 [[x, y, z], ...]。
    tau : int or None
        时间延迟。None 时用互信息法自动计算。
    m : int or None
        嵌入维数。None 时用 FNN 法自动计算。
    dt : float or None
        采样间隔（秒）。None 时 λ 单位为 nat/step；
        提供时 λ 单位为 nat/s（与论文一致）。
    p_period : int
        Theiler 窗口大小，排除时间上过近的伪近邻。
    subject_label : str
        日志前缀。
    show_progress : bool
        是否显示 tqdm 进度条。

    Returns
    -------
    mle : float
        最大 Lyapunov 指数。失败返回 np.nan。
    """
    points = np.asarray(points, dtype=float)
    prefix = f"[{subject_label}] " if subject_label else ""
    logger.info(f"{prefix}MLE计算开始，输入点数: {len(points)}")

    if points.ndim != 2 or points.shape[1] != 3:
        logger.warning(f"{prefix}输入数据维度异常，需为 Nx3，当前 shape={points.shape}")
        return np.nan

    if len(points) < 2:
        logger.warning(f"{prefix}有效点数不足，无法计算速度序列")
        return np.nan

    # ── Step 1: 3D 位置 → 1D 合速度 ──
    velocities = np.linalg.norm(np.diff(points, axis=0), axis=1)
    logger.info(f"{prefix}速度序列长度: {len(velocities)}")

    # ── Step 2: DFT 去噪 ──
    if ENABLE_DFT_DENOISE:
        velocities_processed = dft_denoise(velocities)
        logger.info(f"{prefix}DFT 去噪完成")
    else:
        velocities_processed = velocities
        logger.info(f"{prefix}跳过 DFT 去噪，使用原始速度序列")

    # ── Step 3: 自动计算 τ (互信息法) ──
    if tau is None:
        tau = mutual_information(velocities_processed)
        logger.info(f"{prefix}互信息法自动 τ = {tau}")
    else:
        logger.info(f"{prefix}使用指定 τ = {tau}")

    # ── Step 4: 自动计算 m (FNN 法) ──
    if m is None:
        m = false_nearest_neighbors(velocities_processed, tau)
        logger.info(f"{prefix}FNN 法自动 m = {m}")
    else:
        logger.info(f"{prefix}使用指定 m = {m}")

    # ── Step 5: Takens 延迟嵌入 → 相空间重构 ──
    embedded = time_delay_embedding(velocities_processed, tau, m)
    n_points = embedded.shape[0]
    min_required = p_period + 5 + 1
    if n_points < min_required:
        logger.warning(
            f"{prefix}重构后有效点数不足，n_points={n_points}, 至少需要 {min_required}"
        )
        return np.nan

    logger.info(f"{prefix}相空间重构完成: {n_points} 点 × {m} 维")

    # ── Step 6: Rosenstein 近邻散度追踪 ──
    # 追踪步长设为足够大，后续由自适应线性区裁剪
    max_track = min(150, n_points - p_period - 1)

    sum_log_d = np.zeros(max_track + 1, dtype=float)
    count = np.zeros(max_track + 1, dtype=int)
    used_reference_count = 0

    # 候选近邻数阈值：太少则跳过该参考点
    MIN_CANDIDATES = 10

    i_iter = range(p_period, n_points - max_track)
    if show_progress:
        i_iter = tqdm(
            i_iter,
            desc=f"{subject_label or '当前样本'}-MLE",
            unit="step",
            leave=False,
        )

    for i in i_iter:
        ref_point = embedded[i].reshape(1, -1)

        # Theiler 窗口：双向排除 |i - j| <= p_period 的点
        left = np.arange(0, max(0, i - p_period))
        right = np.arange(i + p_period, n_points - max_track)
        candidate_indices = np.concatenate([left, right])
        if len(candidate_indices) < MIN_CANDIDATES:
            continue

        candidates = embedded[candidate_indices]
        distances = cdist(ref_point, candidates)[0]

        j_local = int(np.argmin(distances))
        j_origin = int(candidate_indices[j_local])

        used_reference_count += 1
        for k in range(max_track + 1):
            d_k = np.linalg.norm(embedded[i + k] - embedded[j_origin + k])
            if d_k < 1e-12:
                continue
            sum_log_d[k] += np.log(d_k)
            count[k] += 1

    valid_mask = count > 0
    if not np.any(valid_mask):
        logger.warning(f"{prefix}无有效演化轨迹，返回 NaN")
        return np.nan

    valid_k = np.where(valid_mask)[0]
    y_avg = sum_log_d[valid_k] / count[valid_k]

    # ── Step 7: 自适应线性区识别 + 回归 ──
    linear_end = find_linear_region(y_avg, min_points=5)
    k_fit = valid_k[:linear_end + 1]
    y_fit = y_avg[:linear_end + 1]

    if len(k_fit) < 3:
        logger.warning(f"{prefix}线性区过短(仅{len(k_fit)}点)，返回 NaN")
        return np.nan

    try:
        reg = LinearRegression().fit(k_fit.reshape(-1, 1), y_fit)
        mle_raw = float(reg.coef_[0])  # 单位: nat/step

        # ── Step 8: 除以 Δt → nat/s ──
        if dt is not None and dt > 0:
            mle_value = mle_raw / dt
        else:
            mle_value = mle_raw

        logger.info(
            f"{prefix}MLE计算完成，参考点数={used_reference_count}, "
            f"有效k={len(valid_k)}, 线性区k=[{k_fit[0]}, {k_fit[-1]}], "
            f"τ={tau}, m={m}, "
            f"MLE={mle_value:.8f} nat/{'s' if dt else 'step'}"
        )
        return mle_value
    except ValueError:
        logger.warning(f"{prefix}线性回归失败，返回 NaN")
        return np.nan


# ═══════════════════════════════════════════════════════════════
# 以下：数据读取 / 文件发现 / 并行计算 / 标准结构输出（与其他指标统一） / 统计检验
# （MLE 计算核心与原版一致；数据层微调了 dt 传递，并新增标准结构输出）
# ═══════════════════════════════════════════════════════════════

def read_relative_points(csv_path):
    try:
        df = pd.read_csv(csv_path, index_col=False)
    except Exception as exc:
        print(f"读取失败 {csv_path}: {exc}")
        return np.empty((0, 3), dtype=float)

    cols = list(df.columns)
    use_cols = ["LocalPosX", "LocalPosY", "LocalPosZ"]
    if not all(col in cols for col in use_cols):
        print(f"警告: {csv_path} 缺少 LocalPosX/LocalPosY/LocalPosZ 列，跳过")
        return np.empty((0, 3), dtype=float)

    data = df[use_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if data.size == 0:
        return np.empty((0, 3), dtype=float)

    data = data[np.all(np.isfinite(data), axis=1)]
    return data


def discover_scene_subject_files():
    """
    遍历所有场景目录，为每个受试者找到其 _3D 输出的 RelativePosInfo.csv。
    返回 list of dict: {"scene": scene_name, "subject": subject_id, "path": csv_path}
    """
    file_infos = []
    for scene_folder in SCENARIO_FOLDERS:
        scene_rel_dir = os.path.join(RELATIVE_POS_BASE_DIR, scene_folder)
        if not os.path.isdir(scene_rel_dir):
            continue
        for subject_id in os.listdir(scene_rel_dir):
            subject_dir = os.path.join(scene_rel_dir, subject_id)
            if not os.path.isdir(subject_dir):
                continue
            csv_path = os.path.join(subject_dir, "RelativePosInfo.csv")
            if os.path.exists(csv_path):
                file_infos.append({
                    "scene": scene_folder,
                    "subject": subject_id,
                    "path": csv_path,
                })
    return file_infos


def _process_one_subject(args):
    """并行 worker: 读文件 → 算 MLE。args = (scene, subject, csv_path, dt)."""
    scene, subject, csv_path, dt = args
    points = read_relative_points(csv_path)
    point_count = len(points)
    if point_count > 0:
        mle = get_lyapunov_rosenstein(
            points,
            dt=dt,
            subject_label=f"{scene}/{subject}",
            show_progress=False,
        )
    else:
        mle = np.nan
    return {
        "场景": scene,
        "受试者": subject,
        "轨迹点数量": point_count,
        "MLE": mle,
        "数据路径": csv_path,
    }


# ═══════════════════════════════════════════════════════════════
# 标准结构输出（与其他数值指标统一）
#   每场景 <场景>/LyapunovMLE.csv + 顶层 AllSubjects_LyapunovMLE.csv，
#   列结构与 INDICATORS/Results_Evacuation* 系列保持一致，便于后续标准化分析
# ═══════════════════════════════════════════════════════════════

STANDARD_COLUMNS = ["Scenario", "Subject", "FileName", "MLE", "TotalPoints"]


def build_standard_table(individual_stats):
    """将个人级统计（中文列）转换为与其他数值指标统一的英文列结构。"""
    rows = []
    for r in individual_stats:
        rows.append({
            "Scenario": r["场景"],
            "Subject": r["受试者"],
            "FileName": os.path.basename(str(r["数据路径"])),
            "MLE": r["MLE"],
            "TotalPoints": r["轨迹点数量"],
        })
    return pd.DataFrame(rows, columns=STANDARD_COLUMNS)


def save_standard_outputs(individual_stats, out_base_dir):
    """
    按其他数值指标（INDICATORS/Results_Evacuation*）的目录与列约定输出：
      - <out_base_dir>/<场景>/LyapunovMLE.csv        每场景一份
      - <out_base_dir>/AllSubjects_LyapunovMLE.csv   全量汇总（场景按 SCENARIO_FOLDERS 顺序排列）
    """
    df_std = build_standard_table(individual_stats)

    scene_frames = []
    for scene_folder in SCENARIO_FOLDERS:
        scene_df = (
            df_std[df_std["Scenario"] == scene_folder]
            .sort_values("Subject")
        )
        if scene_df.empty:
            continue
        scene_dir = os.path.join(out_base_dir, scene_folder)
        os.makedirs(scene_dir, exist_ok=True)
        scene_path = os.path.join(scene_dir, "LyapunovMLE.csv")
        scene_df.to_csv(scene_path, index=False, encoding="utf-8-sig")
        logger.info(f"  场景结果已保存: {scene_path} ({len(scene_df)} 行)")
        scene_frames.append(scene_df)

    if scene_frames:
        overall_df = pd.concat(scene_frames, ignore_index=True)
        overall_path = os.path.join(out_base_dir, "AllSubjects_LyapunovMLE.csv")
        overall_df.to_csv(overall_path, index=False, encoding="utf-8-sig")
        logger.info(f"总体结果已保存: {overall_path} ({len(overall_df)} 行)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("Rosenstein MLE 分析脚本 (适配 Experiment_Data_A5)")
    logger.info(f"工程根目录: {PROJECT_ROOT}")
    logger.info(f"输出目录: {OUTPUT_DIR}")

    # VR 数据帧间隔 (50Hz)
    DT = 1.0 / 50

    # ── Step 1: 发现所有受试者的 3D RelativePosInfo ──
    file_infos = discover_scene_subject_files()
    logger.info(f"发现 {len(file_infos)} 个 RelativePosInfo.csv 文件")

    if len(file_infos) == 0:
        logger.error("未找到任何数据文件，请先运行 ExtractRelativePos_3D.py")
        return

    # 按场景统计受试者数量
    scene_counts = {}
    for info in file_infos:
        s = info["scene"]
        scene_counts[s] = scene_counts.get(s, 0) + 1
    for s, n in scene_counts.items():
        logger.info(f"  {s}: {n} 名受试者")

    # ── Step 2: 并行计算 MLE ──
    tasks = [
        (info["scene"], info["subject"], info["path"], DT)
        for info in file_infos
    ]
    logger.info(f"启动并行 MLE 计算 (max_workers=20)")

    individual_stats = []
    with ProcessPoolExecutor(max_workers=20) as executor:
        future_map = {executor.submit(_process_one_subject, t): t for t in tasks}
        overall_iter = tqdm(
            as_completed(future_map),
            total=len(tasks),
            desc="MLE-Rosenstein",
            unit="受试者",
        )
        for future in overall_iter:
            result = future.result()
            individual_stats.append(result)
            overall_iter.set_postfix_str(f"{result['场景']}/{result['受试者']}")

    # ── Step 3: 输出标准结构结果表（与其他数值指标统一） ──
    save_standard_outputs(individual_stats, OUTPUT_DIR)

    # ── Step 4: 输出个人级统计表 ──
    df_individual = pd.DataFrame(individual_stats).sort_values(["场景", "受试者"])
    df_individual.to_csv(
        os.path.join(OUTPUT_DIR, "个人MLE统计表.csv"),
        index=False, encoding="utf-8-sig"
    )
    logger.info(f"个人MLE统计表已保存 ({len(df_individual)} 行)")

    # ── Step 5: 按场景汇总 ──
    summary_rows = []
    for scene_folder in SCENARIO_FOLDERS:
        scene_data = df_individual[df_individual["场景"] == scene_folder]["MLE"].dropna()
        n_val = len(scene_data)
        if n_val == 0:
            summary_rows.append({
                "场景": scene_folder, "有效样本量": 0,
                "均值": "", "标准差": "", "最小值": "", "最大值": "",
            })
            continue
        summary_rows.append({
            "场景": scene_folder,
            "有效样本量": n_val,
            "均值": f"{np.mean(scene_data):.8f}",
            "标准差": f"{np.std(scene_data, ddof=1):.8f}" if n_val > 1 else "",
            "最小值": f"{np.min(scene_data):.8f}",
            "最大值": f"{np.max(scene_data):.8f}",
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(
        os.path.join(OUTPUT_DIR, "场景MLE汇总表.csv"),
        index=False, encoding="utf-8-sig"
    )
    logger.info("场景MLE汇总表已保存")

    # ── Step 6: 正态性检验 ──
    normality_rows = []
    for scene_folder in SCENARIO_FOLDERS:
        scene_data = df_individual[df_individual["场景"] == scene_folder]["MLE"].dropna()
        n_val = len(scene_data)
        if n_val >= 3:
            stat, p_value = stats.shapiro(scene_data)
            normality_rows.append({
                "场景": scene_folder, "样本量": n_val,
                "Shapiro-W": f"{stat:.4f}", "p值": f"{p_value:.4f}",
                "正态性": "正态" if p_value > 0.05 else "非正态",
            })
        else:
            normality_rows.append({
                "场景": scene_folder, "样本量": n_val,
                "Shapiro-W": "N/A", "p值": "N/A",
                "正态性": "样本不足(<3)" if n_val > 0 else "无数据",
            })

    df_normality = pd.DataFrame(normality_rows)
    df_normality.to_csv(
        os.path.join(OUTPUT_DIR, "正态性检验结果.csv"),
        index=False, encoding="utf-8-sig"
    )
    logger.info("正态性检验结果已保存")

    # ── Step 7: 两两对比（关键对比：同烟雾不同干预 / 同干预不同烟雾）──
    scene_mle_map = {}
    for _, row in df_individual.iterrows():
        s = row["场景"]
        if pd.notna(row["MLE"]):
            scene_mle_map.setdefault(s, []).append(row["MLE"])

    # 构造有意义的对比对
    interventions = ["StaticSign", "ArrowRibbon", "MiniMap", "TextBar", "Full"]
    smoke_levels = ["HighSmoke", "LowSmoke"]

    comparison_rows = []

    # 对比1: 同烟雾不同干预
    for smoke in smoke_levels:
        for i in range(len(interventions)):
            for j in range(i + 1, len(interventions)):
                g1 = f"{interventions[i]}-{smoke}"
                g2 = f"{interventions[j]}-{smoke}"
                d1 = scene_mle_map.get(g1, [])
                d2 = scene_mle_map.get(g2, [])
                if len(d1) < 3 or len(d2) < 3:
                    continue
                stat, pv = stats.mannwhitneyu(d1, d2, alternative="two-sided")
                comparison_rows.append({
                    "对比类型": "同烟雾-不同干预",
                    "组1": g1, "组2": g2,
                    "组1样本量": len(d1), "组2样本量": len(d2),
                    "组1均值": f"{np.mean(d1):.8f}", "组2均值": f"{np.mean(d2):.8f}",
                    "检验方法": "Mann-Whitney U",
                    "统计量": f"{stat:.4f}", "p值": f"{pv:.4f}",
                    "显著性": "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns")),
                })

    # 对比2: 同干预不同烟雾
    for interv in interventions:
        g1 = f"{interv}-HighSmoke"
        g2 = f"{interv}-LowSmoke"
        d1 = scene_mle_map.get(g1, [])
        d2 = scene_mle_map.get(g2, [])
        if len(d1) < 3 or len(d2) < 3:
            continue
        # 配对样本优先用 Wilcoxon（同一批受试者）
        if len(d1) == len(d2):
            stat, pv = stats.wilcoxon(d1, d2)
            method = "Wilcoxon"
        else:
            stat, pv = stats.mannwhitneyu(d1, d2, alternative="two-sided")
            method = "Mann-Whitney U (样本量不等)"
        comparison_rows.append({
            "对比类型": "同干预-不同烟雾",
            "组1": g1, "组2": g2,
            "组1样本量": len(d1), "组2样本量": len(d2),
            "组1均值": f"{np.mean(d1):.8f}", "组2均值": f"{np.mean(d2):.8f}",
            "检验方法": method,
            "统计量": f"{stat:.4f}", "p值": f"{pv:.4f}",
            "显著性": "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns")),
        })

    # 对比3: 全局 Kruskal-Wallis（10组整体差异）
    all_groups_kw = [v for v in scene_mle_map.values() if len(v) >= 3]
    if len(all_groups_kw) >= 3:
        kw_stat, kw_pv = stats.kruskal(*all_groups_kw)
        comparison_rows.append({
            "对比类型": "全局检验",
            "组1": "全部10组", "组2": "",
            "组1样本量": "", "组2样本量": "",
            "组1均值": "", "组2均值": "",
            "检验方法": "Kruskal-Wallis",
            "统计量": f"{kw_stat:.4f}", "p值": f"{kw_pv:.4f}",
            "显著性": "***" if kw_pv < 0.001 else ("**" if kw_pv < 0.01 else ("*" if kw_pv < 0.05 else "ns")),
        })

    df_comparison = pd.DataFrame(comparison_rows)
    df_comparison.to_csv(
        os.path.join(OUTPUT_DIR, "两两对比检验结果.csv"),
        index=False, encoding="utf-8-sig"
    )
    logger.info(f"两两对比检验结果已保存 ({len(comparison_rows)} 组对比)")

    # ── 打印关键结果到控制台 ──
    print("\n" + "=" * 70)
    print("场景级 MLE 汇总")
    print("=" * 70)
    print(df_summary.to_string(index=False))

    print("\n" + "=" * 70)
    print("显著性标注: *** p<0.001  ** p<0.01  * p<0.05  ns 不显著")
    print("=" * 70)
    print(df_comparison.to_string(index=False))

    print("\n" + "=" * 70)
    print(f"全部结果已保存至: {OUTPUT_DIR}")
    print("  - AllSubjects_LyapunovMLE.csv + <场景>/LyapunovMLE.csv（标准结构，与其他数值指标统一）")
    print("  - 个人MLE统计表.csv")
    print("  - 场景MLE汇总表.csv")
    print("  - 正态性检验结果.csv")
    print("  - 两两对比检验结果.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
