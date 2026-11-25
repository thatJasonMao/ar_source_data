#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一的实测轨迹语义分区接口（Unity XZ 水平面、Y 垂直轴）。

复用 SemanticZoneQuery 的正式 JSON 几何、优先级和跨层楼梯规则。
仅分类时将 [split_y - 0.05, split_y) 内的 Y 吸附到 split_y + 0.0001，
与 Extract_ZoneTimeline 已验证规则一致。此窄带是容差约定，而非真实位置重建。
不修改输入数组，不过滤或重排行，不改时间戳；数据清洗仍由调用方负责。
无效或超出几何范围的点返回 Null / -1。

使用：
    query = UnifiedZoneQuery()
    zone_id = query.classify_point(x, y, z)
    classifier = VectorizedZoneClassifier(query)
    zone_indices = classifier.assign(original_xyz)
    n_snapped = query.count_floor_drift(original_xyz)

坐标输出与距离计算必须继续使用 original_xyz。
本模块只依赖 numpy 和原始几何查询模块，不依赖绘图或提取脚本。
"""

from __future__ import annotations

import numpy as np

from SemanticZoneQuery import SemanticZoneQuery, DEFAULT_GEOMETRY_PATH

DEFAULT_FLOOR_DRIFT_TOLERANCE = 0.05
FLOOR_SNAP_OFFSET = 1e-4


def floor_drift_mask(y, split_y, tolerance=DEFAULT_FLOOR_DRIFT_TOLERANCE):
    """与时间线原版吸附区间完全相同，左闭右开；不修改输入。"""
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("floor drift tolerance must be finite and nonnegative")
    y = np.asarray(y, dtype=np.float64)
    return (y >= split_y - tolerance) & (y < split_y)


def corrected_y(y, split_y, tolerance=DEFAULT_FLOOR_DRIFT_TOLERANCE):
    """返回仅用于分类的高度副本，支持标量和数组。"""
    return np.where(floor_drift_mask(y, split_y, tolerance),
                    split_y + FLOOR_SNAP_OFFSET, y)


def classification_positions(points, split_y, tolerance=DEFAULT_FLOOR_DRIFT_TOLERANCE):
    """复制 (N, 3) 坐标并修正 Y，保留原始数组及帧对应关系。"""
    result = np.array(points, dtype=np.float64, copy=True)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    result[:, 1] = corrected_y(result[:, 1], split_y, tolerance)
    return result


class UnifiedZoneQuery(SemanticZoneQuery):
    """带实测地板高度容差的单点查询；原始几何模块保持独立。

    继承 classify_point/classify_position/classify_many/__call__，
    这些方法均经由本类 find_matches 使用统一修正。
    """

    def __init__(self, geometry_path=DEFAULT_GEOMETRY_PATH, *,
                 floor_drift_tolerance=DEFAULT_FLOOR_DRIFT_TOLERANCE, **kwargs):
        floor_drift_mask([], 0.0, floor_drift_tolerance)  # 校验容差
        self.floor_drift_tolerance = float(floor_drift_tolerance)
        super().__init__(geometry_path, **kwargs)

    def classify_floor(self, y):
        if not np.isfinite(y):
            return None
        y = float(corrected_y(y, self.split_y, self.floor_drift_tolerance))
        return super().classify_floor(y)

    def find_matches(self, x, y, z):
        if not np.isfinite([x, y, z]).all():
            return ()
        y = float(corrected_y(y, self.split_y, self.floor_drift_tolerance))
        return super().find_matches(x, y, z)

    def count_floor_drift(self, points):
        """统计吸附帧，供时间线输出 n_snapped_frames；不修改坐标。"""
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        return int(floor_drift_mask(points[:, 1], self.split_y,
                                    self.floor_drift_tolerance).sum())


class VectorizedZoneClassifier:
    """批量 Unity 分区查询；与 UnifiedZoneQuery 单点查询使用同一高度修正。

    assign 输入 (N, 3) 原始 XYZ，内部复制并修正分类坐标，返回区域索引。
    区域排序、跨层楼梯、XZ/Y collider 与边界仲裁沿用正式几何定义。
    保留 query/zone_list/global_index/floor_of 接口供既有调用方使用。
    """

    def __init__(self, query: UnifiedZoneQuery | None = None):
        query = UnifiedZoneQuery() if query is None else query
        self.query = query
        self.floor_drift_tolerance = getattr(query, "floor_drift_tolerance", DEFAULT_FLOOR_DRIFT_TOLERANCE)
        self.eps = query.epsilon
        self.zone_list = list(query.zones)
        self.global_index = {z.zone_id: i for i, z in enumerate(self.zone_list)}
        self._floor_tables = {}
        for floor in (query.PLATFORM, query.HALL):
            zlist = [z for z in self.zone_list if z.belongs_to_floor(floor)]
            rows = []
            for zi, z in enumerate(zlist):
                for p in z.collider_parts:
                    rows.append((zi, p.xmin, p.xmax, p.zmin, p.zmax, z.y_min, z.y_max))
            arr = np.asarray(rows, dtype=np.float64)  # (n_parts, 7)
            part_zone = arr[:, 0].astype(np.int64)
            # 与 SemanticZoneQuery 一致的仲裁排序键：(-priority, area_xz, zone_id)
            keys = [(-z.priority, z.area_xz, z.zone_id) for z in zlist]
            order = sorted(range(len(zlist)), key=lambda i: keys[i])
            rank = np.empty(len(zlist), dtype=np.int64)
            for r, zi in enumerate(order):
                rank[zi] = r
            self._floor_tables[floor] = (zlist, arr, part_zone, rank)

    def floor_of(self, y: np.ndarray) -> np.ndarray:
        """返回每个点的楼层编码：0=Platform, 1=Hall, -1=越界 Null。"""
        q = self.query
        y = corrected_y(y, q.split_y, self.floor_drift_tolerance)
        out = np.full(len(y), -1, dtype=np.int8)
        valid = (y >= q.platform_floor_y - self.eps) & (y <= q.hall_ceiling_y + self.eps)
        out[valid & (y < q.split_y)] = 0
        out[valid & (y >= q.split_y)] = 1
        return out

    def assign(self, points: np.ndarray, chunk: int = 50000) -> np.ndarray:
        """将 (N, 3) 点集映射为 self.zone_list 的全局索引数组（-1 = Null）。"""
        if chunk <= 0:
            raise ValueError("chunk must be positive")
        points = classification_positions(points, self.query.split_y, self.floor_drift_tolerance)
        n = len(points)
        result = np.full(n, -1, dtype=np.int64)
        floors = self.floor_of(points[:, 1])
        for fi, floor in enumerate((self.query.PLATFORM, self.query.HALL)):
            idx = np.nonzero(floors == fi)[0]
            if len(idx) == 0:
                continue
            zlist, arr, part_zone, rank = self._floor_tables[floor]
            nz = len(zlist)
            xmin, xmax = arr[:, 1], arr[:, 2]
            zmin, zmax = arr[:, 3], arr[:, 4]
            ymin, ymax = arr[:, 5], arr[:, 6]
            for s in range(0, len(idx), chunk):
                chunk_idx = idx[s:s + chunk]
                pts = points[chunk_idx]
                inside = (
                    (pts[:, 0, None] >= xmin[None, :] - self.eps)
                    & (pts[:, 0, None] <= xmax[None, :] + self.eps)
                    & (pts[:, 2, None] >= zmin[None, :] - self.eps)
                    & (pts[:, 2, None] <= zmax[None, :] + self.eps)
                    & (pts[:, 1, None] >= ymin[None, :] - self.eps)
                    & (pts[:, 1, None] <= ymax[None, :] + self.eps)
                )
                # parts -> zones（任一部件命中即命中该 zone）
                hit = np.zeros((len(chunk_idx), nz), dtype=bool)
                for zi in range(nz):
                    part_mask = part_zone == zi
                    hit[:, zi] = inside[:, part_mask].any(axis=1)
                # 取仲裁排序中 rank 最小（最优先）的命中分区
                ranked = np.where(hit, rank[None, :], np.iinfo(np.int64).max)
                best = np.argmin(ranked, axis=1)
                ok = ranked[np.arange(len(chunk_idx)), best] != np.iinfo(np.int64).max
                if ok.any():
                    result[chunk_idx[ok]] = [
                        self.global_index[zlist[b].zone_id] for b in best[ok]
                    ]
        return result

