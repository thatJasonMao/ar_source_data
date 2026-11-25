"""Direct planar isovists; smoke limits geometry rays, never a lookup field."""
from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import shapely
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from UnifiedZoneQuery import UnifiedZoneQuery
from SemanticZoneQuery import DEFAULT_GEOMETRY_PATH

HIGH_SMOKE_VISIBILITY_M = 7.5
LOW_SMOKE_VISIBILITY_M = 12.5
SMOKE = {'high': HIGH_SMOKE_VISIBILITY_M, 'low': LOW_SMOKE_VISIBILITY_M}
METRICS = {
    'horizontal_size': ('Horizontal size', 'm$^2$'),
    'concavity': ('Concavity', '1 - A / A_hull'),
    'distance_variation': ('Distance variation', 'SD(r) / mean(r)'),
    'elongation': ('Elongation', 'sqrt(lambda_max / lambda_min)'),
}
STAIRS = frozenset({'Stair_ABCD', 'Stair_EFGH'})

@dataclass(frozen=True)
class IsovistResult:
    status: str
    floor: str | None
    visibility_m: float
    horizontal_size: float = float('nan')
    concavity: float = float('nan')
    distance_variation: float = float('nan')
    elongation: float = float('nan')


def polygon_metrics(vertices, radii):
    """Area-weighted polygon moments, NOT covariance of boundary vertices."""
    p = np.asarray(vertices, dtype=float)
    q = np.roll(p, -1, axis=0)
    cross = p[:, 0]*q[:, 1] - q[:, 0]*p[:, 1]
    area = cross.sum()/2
    if area < 0:
        return polygon_metrics(p[::-1], radii)
    if area <= 1e-14:
        return (float('nan'),)*4
    centroid = ((p+q)*cross[:, None]).sum(axis=0)/(6*area)
    second = ((p*p+p*q+q*q)*cross[:, None]).sum(axis=0)/(12*area)
    mixed = ((2*p[:, 0]*p[:, 1]+p[:, 0]*q[:, 1]+q[:, 0]*p[:, 1]
              +2*q[:, 0]*q[:, 1])*cross).sum()/(24*area)
    cov = np.array([[second[0], mixed], [mixed, second[1]]])-np.outer(centroid, centroid)
    ev = np.linalg.eigvalsh(cov)
    elongation = float(np.sqrt(ev[1]/ev[0])) if ev[0] > 1e-14 else float('nan')
    hull_area = Polygon(p).convex_hull.area
    concavity = float(np.clip(1-area/hull_area, 0, 1))
    mean = np.mean(radii)
    variation = float(np.std(radii, ddof=0)/mean) if mean > 0 else float('nan')
    return float(area), concavity, variation, elongation


class PlanarIsovist:
    """Immutable floor geometry plus direct queries at arbitrary Unity positions.

    Y only selects a floor through UnifiedZoneQuery (including its drift fix).
    An explicit floor accepts y=None, avoiding an invented vertical filter.
    Stair footprints exclude origins only, never block rays.
    """
    def __init__(self, geometry_path=DEFAULT_GEOMETRY_PATH, *, rays=720):
        if isinstance(rays, bool) or int(rays) != rays or rays < 32:
            raise ValueError('rays must be an integer >= 32')
        self.rays = int(rays)
        self.geometry_path = Path(geometry_path).resolve()
        self.zone_query = UnifiedZoneQuery(self.geometry_path)
        self.domains, self.exclusions, self.segments = {}, {}, {}
        for floor in ('Hall', 'Platform'):
            zones = [z for z in self.zone_query.zones if z.belongs_to_floor(floor)]
            def union(items):
                return unary_union([box(p.xmin, p.zmin, p.xmax, p.zmax)
                                    for z in items for p in z.collider_parts])
            domain = union(zones)
            if domain.is_empty or not domain.is_valid:
                raise ValueError(f'Invalid {floor} boundary')
            self.domains[floor] = domain
            self.exclusions[floor] = union([z for z in zones if z.zone_id in STAIRS])
            starts, ends = [], []
            for part in getattr(domain, 'geoms', [domain]):
                for ring in (part.exterior, *part.interiors):
                    coords = np.asarray(ring.coords)
                    starts.extend(coords[:-1]); ends.extend(coords[1:])
            a, b = np.asarray(starts), np.asarray(ends)
            good = np.linalg.norm(b-a, axis=1) > 1e-12
            self.segments[floor] = (a[good], b[good])
        self.angles = np.arange(self.rays)*2*np.pi/self.rays
        self.directions = np.column_stack((np.cos(self.angles), np.sin(self.angles)))

    def locate(self, x, y, z, *, floor=None):
        if not np.isfinite([x, z]).all():
            return None, 'invalid_position'
        if floor is not None and floor not in self.domains:
            raise ValueError('floor must be Hall or Platform')
        if floor is None:
            if y is None or not np.isfinite(y):
                return None, 'invalid_position'
            floor = self.zone_query.classify_floor(float(y))
        if floor is None:
            return None, 'outside_height_envelope'
        point = Point(float(x), float(z))
        if not self.domains[floor].covers(point):
            return floor, 'outside_floor'
        if self.exclusions[floor].covers(point):
            return floor, 'excluded_stair'
        return floor, 'ok'

    def _radial(self, origin, floor, radius, directions, a, b):
        v, w = b-a, a-origin
        den = directions[:, 0, None]*v[None, :, 1]-directions[:, 1, None]*v[None, :, 0]
        parallel = np.abs(den) < 1e-13
        safe_den = np.where(parallel, 1, den)
        t = (w[:, 0]*v[:, 1]-w[:, 1]*v[:, 0])[None, :]/safe_den
        s = (w[None, :, 0]*directions[:, 1, None]-w[None, :, 1]*directions[:, 0, None])/safe_den
        hits = (~parallel) & (t > 1e-10) & (s >= -1e-10) & (s <= 1+1e-10)
        r = np.minimum(np.min(np.where(hits, t, np.inf), axis=1, initial=np.inf), radius)
        # At an exact wall origin only outward rays have zero length.
        if self.domains[floor].boundary.distance(Point(*origin)) < 1e-10:
            ahead = origin+np.where(np.abs(directions) < 1e-12, 0.0, directions)*1e-8
            inside = shapely.intersects_xy(self.domains[floor], ahead[:, 0], ahead[:, 1])
            r[~inside] = 0
        return r

    def query(self, x, y, z, visibility_m, *, floor=None):
        radius = float(visibility_m)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError('visibility_m must be positive and finite')
        floor, status = self.locate(x, y, z, floor=floor)
        if status != 'ok':
            return IsovistResult(status, floor, radius)
        origin = np.array([x, z], dtype=float)
        a, b = self.segments[floor]
        # Bounding-box cull retains every segment that can meet the radius disk.
        keep = np.all(np.maximum(a, b) >= origin-radius-1e-10, axis=1) & np.all(np.minimum(a, b) <= origin+radius+1e-10, axis=1)
        a, b = a[keep], b[keep]
        radii = self._radial(origin, floor, radius, self.directions, a, b)
        # Add corner-sided rays for polygon fidelity, but never include these
        # nonuniform samples in the radial coefficient of variation.
        vertices = np.unique(np.concatenate((a, b)), axis=0) if len(a) else np.empty((0, 2))
        delta = vertices-origin
        near = (np.linalg.norm(delta, axis=1) <= radius) & (np.linalg.norm(delta, axis=1) > 1e-10)
        corner_angles = np.arctan2(delta[near, 1], delta[near, 0])
        extra = np.mod((corner_angles[:, None]+np.array([-1e-8, 0, 1e-8])).ravel(), 2*np.pi)
        if len(extra):
            directions = np.column_stack((np.cos(extra), np.sin(extra)))
            extra_r = self._radial(origin, floor, radius, directions, a, b)
            order = np.argsort(np.concatenate((self.angles, extra)))
            points = np.concatenate((self.directions*radii[:, None], directions*extra_r[:, None]))[order]
        else:
            points = self.directions*radii[:, None]
        values = polygon_metrics(points, radii)
        return IsovistResult('ok' if np.isfinite(values).all() else 'degenerate_isovist', floor, radius, *values)

    def query_high_smoke(self, x, y, z, *, floor=None):
        return self.query(x, y, z, HIGH_SMOKE_VISIBILITY_M, floor=floor)

    def query_low_smoke(self, x, y, z, *, floor=None):
        return self.query(x, y, z, LOW_SMOKE_VISIBILITY_M, floor=floor)


@lru_cache(maxsize=1)
def get_engine():
    """Cache geometry only; every result is computed from the requested position."""
    return PlanarIsovist()


def _outline(ax, geometry, **kwargs):
    for p in getattr(geometry, 'geoms', [geometry]):
        if p.is_empty:
            continue
        for ring in (p.exterior, *p.interiors):
            c = -np.asarray(ring.coords)  # same 180-degree display as semantic map
            ax.plot(c[:, 0], c[:, 1], **kwargs)


def render_heatmaps(engine, metrics, output, *, cell=0.4, dpi=300, force=False):
    """Sampling is for images only. No query API loads these image samples."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from tqdm import tqdm
    output = Path(output)
    if not np.isfinite(cell) or cell <= 0:
        raise ValueError('cell must be positive')
    if any(m not in METRICS for m in metrics):
        raise ValueError('Unknown metric')
    targets = [output/f'{m}_{s}.png' for m in metrics for s in SMOKE]
    manifest_path = output/('manifest_'+ '_'.join(metrics)+'.json')
    if not force and any(p.exists() for p in targets+[manifest_path]):
        raise FileExistsError('Output exists; use --force to regenerate it')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    grids, counts = {}, {}
    for floor, domain in engine.domains.items():
        xmin, zmin, xmax, zmax = domain.bounds
        xe = xmin+np.arange(int(np.ceil((xmax-xmin)/cell))+1)*cell
        ze = zmin+np.arange(int(np.ceil((zmax-zmin)/cell))+1)*cell
        xx, zz = np.meshgrid((xe[:-1]+xe[1:])/2, (ze[:-1]+ze[1:])/2)
        valid = shapely.intersects_xy(domain, xx, zz) & ~shapely.intersects_xy(engine.exclusions[floor], xx, zz)
        indices = np.argwhere(valid)
        counts[floor] = int(valid.sum())
        values = {s: {m: np.full(xx.shape, np.nan) for m in metrics} for s in SMOKE}
        for iz, ix in tqdm(indices, desc=f'{floor}: direct isovists'):
            for smoke, radius in SMOKE.items():
                result = engine.query(xx[iz, ix], None, zz[iz, ix], radius, floor=floor)
                if result.status != 'ok':
                    raise RuntimeError(f'Unexpected sampled result: {result}')
                for m in metrics:
                    values[smoke][m][iz, ix] = getattr(result, m)
        grids[floor] = (xe, ze, values)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 10})
    scales = {}
    for metric in metrics:
        vals = np.concatenate([g[2][s][metric][np.isfinite(g[2][s][metric])] for g in grids.values() for s in SMOKE])
        vmin = 1.0 if metric == 'elongation' else 0.0
        vmax = float(vals.max())
        if vmax <= vmin:
            vmax = vmin+1e-6
        scales[metric] = [vmin, vmax]
        for smoke, radius in SMOKE.items():
            fig, axes = plt.subplots(1, 2, figsize=(10, 12), layout='constrained', gridspec_kw={'width_ratios': [3, 1]})
            for ax, floor in zip(axes, ('Hall', 'Platform')):
                xe, ze, values = grids[floor]
                im = ax.pcolormesh(-xe[::-1], -ze[::-1], values[smoke][metric][::-1, ::-1], cmap='plasma', norm=Normalize(vmin, vmax), shading='flat', rasterized=True)
                _outline(ax, engine.domains[floor], color='red', linewidth=0.7)
                _outline(ax, engine.exclusions[floor], color='gray', linewidth=0.7, linestyle='--')
                ax.set_aspect('equal')
                ax.set_title(floor)
                ax.set_xlabel('Display X = -Unity X (m)')
                ax.set_ylabel('Display Z = -Unity Z (m)')
                ax.ticklabel_format(useOffset=False, style='plain')
            fig.colorbar(im, ax=axes, shrink=0.75, label=f'{METRICS[metric][0]} ({METRICS[metric][1]})')
            fig.suptitle(f'{METRICS[metric][0]} | {smoke.capitalize()} smoke | visibility {radius:g} m\n360-degree planar isovist; gray dashed regions excluded from sampling')
            fig.savefig(output/f'{metric}_{smoke}.png', dpi=dpi)
            plt.close(fig)
    manifest = {
        'geometry_path': str(engine.geometry_path),
        'geometry_sha256': hashlib.sha256(engine.geometry_path.read_bytes()).hexdigest(),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'smoke_visibility_m': SMOKE, 'uniform_rays': engine.rays,
        'corner_refinement_rad': 1e-8, 'cell_m': cell, 'dpi': dpi,
        'sampling': 'cell-center direct evaluation; no smoothing; no field used by APIs',
        'floors': counts, 'excluded_origin_zones': sorted(STAIRS),
        'stairs_occlude': False, 'internal_semantic_edges_occlude': False,
        'floor_split_y': engine.zone_query.split_y,
        'floor_drift_tolerance_m': engine.zone_query.floor_drift_tolerance,
        'y_filter': 'UnifiedZoneQuery floor envelope only; explicit floor bypasses Y',
        'metrics': list(metrics), 'shared_color_scales': scales,
        'elapsed_seconds': time.perf_counter()-started,
        'files': [str(p) for p in targets],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def metric_cli(metric=None):
    parser = argparse.ArgumentParser(description='Direct planar isovist metrics and smoke heatmaps')
    parser.add_argument('--geometry', type=Path, default=DEFAULT_GEOMETRY_PATH)
    parser.add_argument('--output', type=Path, default=HERE/'Results_Planar_Isovists')
    parser.add_argument('--rays', type=int, default=720)
    parser.add_argument('--cell', type=float, default=0.4)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--point', nargs=3, type=float, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--floor', choices=['Hall', 'Platform'])
    args = parser.parse_args()
    engine = PlanarIsovist(args.geometry, rays=args.rays)
    if args.point:
        for smoke, radius in SMOKE.items():
            r = engine.query(*args.point, radius, floor=args.floor)
            print(json.dumps({'smoke': smoke, **asdict(r)}, ensure_ascii=False))
    else:
        render_heatmaps(engine, [metric] if metric else list(METRICS), args.output, cell=args.cell, dpi=args.dpi, force=args.force)

if __name__ == '__main__':
    metric_cli()

