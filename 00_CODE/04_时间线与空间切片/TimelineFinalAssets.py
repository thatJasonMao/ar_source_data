#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproducible, title-free P1-P5 assets for the spatiotemporal figures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from tqdm import tqdm

# Baseline is the packaged assets, whose labels/ticks were already 1.5 times
# the ordinary plots (14/11 pt). Do not apply the new factors to 14/11 pt.
BASE_LABEL_PT = 21.0
BASE_TICK_PT = 16.5
FONT_SCALE = 1.75
MEAN_WIDTH_PT = 3.0 * 1.5
TRIAL_WIDTH_PT = 0.7 * 1.3
SPINE_WIDTH_PT = 0.8 * 2.0
# Smallest common pads (0.5/1 pt increments) passing the 300-dpi text-box
# check for all four metrics, including the two zero ticks at the origin.
LABEL_PAD_PT = 2.5
TICK_PAD_PT = 12.0
BAND_PERCENTILES = (10.0, 90.0)
BAND_COLOR = '#6e7880'
BAND_ALPHA = .28
ASSET_DIRECTORY = '时空动态图素材'
METRICS = {
    'speed': (2.0, 1.0, 'm/s'),
    'gaze_depth': (40.0, 10.0, 'm'),
    'head_search': (100.0, 50.0, 'deg/s'),
    'scan_speed': (125.0, 40.0, 'deg/s'),
}


def draw_trial_quantile_band(ax, centers, trial_lines):
    """Restore the original pooled-trial P10-P90 band; no smoothing or CI.

    The translucent layer sits above individual trials and below group means,
    matching the archived final assets from the 2026-11-02 band update.
    """
    lower = np.full(len(centers), np.nan)
    upper = np.full(len(centers), np.nan)
    if trial_lines:
        matrix = np.asarray([curve for _, _, curve in trial_lines], dtype=float)
        valid = np.isfinite(matrix).any(axis=0)
        matrix = np.where(np.isfinite(matrix), matrix, np.nan)
        lower[valid], upper[valid] = np.nanpercentile(
            matrix[:, valid], BAND_PERCENTILES, axis=0)
        ax.fill_between(centers, lower, upper, color=BAND_COLOR,
                        alpha=BAND_ALPHA, linewidth=0, zorder=2)
    return lower, upper


def _layout_issues(fig, ax):
    """Measure the actual raster renderer, including cross-axis tick collisions."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [ax.xaxis.label, ax.yaxis.label,
             *ax.get_xticklabels(), *ax.get_yticklabels()]
    boxes = [(t.get_text(), t.get_window_extent(renderer)) for t in texts
             if t.get_visible() and t.get_text()]
    gap_px = 2.0 * fig.dpi / 72.0
    issues = []
    for i, (name, box) in enumerate(boxes):
        for other, other_box in boxes[i + 1:]:
            if box.padded(gap_px / 2).overlaps(other_box.padded(gap_px / 2)):
                issues.append(f'{name!r} overlaps {other!r}')
        if (box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width
                or box.y1 > fig.bbox.height):
            issues.append(f'{name!r} outside canvas')
    return issues


def _set_layout(fig, ax, scale):
    ax.set_xlabel('Normalized Time', fontsize=BASE_LABEL_PT * scale, labelpad=LABEL_PAD_PT)
    ax.set_ylabel('Value', fontsize=BASE_LABEL_PT * scale, labelpad=LABEL_PAD_PT)
    ax.tick_params(axis='both', labelsize=BASE_TICK_PT * scale,
                   pad=TICK_PAD_PT, width=SPINE_WIDTH_PT, length=7.0)
    # Two passes settle label extents after the axes position changes.
    fig.tight_layout(pad=.35)
    fig.tight_layout(pad=.35)
    return _layout_issues(fig, ax)


def plot_final_assets(trials, curves, output, centers, dpi, metric):
    """Write only final P1-P5 PNGs and a style/data provenance manifest.

    The ordinary analysis plots keep their own titles, legends and formatting.
    Data and grouping are shared with the four existing analysis entry points.
    """
    from Analyze_TimelineSpeed import (PERIODS, grouping, selected_trial_curves,
        draw_trial_backgrounds, draw_selection_means)

    output = Path(output)
    target = output / ASSET_DIRECTORY
    target.mkdir(parents=True, exist_ok=True)
    ymax, ystep, units = METRICS[metric]
    figures = []
    style = {'font.family':'serif', 'font.serif':['Times New Roman'],
             'font.size':12, 'axes.grid':True, 'grid.alpha':.18,
             'axes.linewidth':SPINE_WIDTH_PT}
    with plt.rc_context(style):
        for pid, _ in tqdm(PERIODS, desc=f'Final assets ({metric})'):
            fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=dpi)
            try:
                trial_lines = selected_trial_curves(trials, curves, pid, 'all', 'AllParticipants')
                lower, upper = draw_trial_quantile_band(ax, centers, trial_lines)
                draw_trial_backgrounds(ax, centers, trial_lines, linewidth=TRIAL_WIDTH_PT)
                means = draw_selection_means(ax, centers, grouping(trials, curves, pid),
                    'all', 'AllParticipants', linewidth=MEAN_WIDTH_PT)
                ax.set(xlim=(0, 1), ylim=(0, ymax))
                ax.set_xticks(np.linspace(0, 1, 6))
                ax.set_yticks(np.arange(0, ymax + 1e-9, ystep))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f' if metric == 'speed' else '%d'))
                ax.spines[['top', 'right']].set_visible(False)
                for spine in ax.spines.values():
                    spine.set_linewidth(SPINE_WIDTH_PT)
                scale = FONT_SCALE
                issues = _set_layout(fig, ax, scale)
                attempts = [{'font_scale':scale, 'issues':issues}]
                if issues:
                    raise ValueError(f'{metric}/{pid}: unresolved layout: {issues}')
                filename = f'{pid}_all_AllParticipants.png'
                path = target / filename
                fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=.025, facecolor='white')
                figures.append({'file':filename, 'period':pid, 'n_trials':len(trial_lines),
                    'n_group_means':len(means), 'font_scale':scale,
                    'axis_label_pt':BASE_LABEL_PT * scale, 'tick_label_pt':BASE_TICK_PT * scale,
                    'figure_inches':fig.get_size_inches().tolist(),
                    'layout_attempts':attempts, 'final_layout_issues':issues,
                    'band_lower':[float(v) if np.isfinite(v) else None for v in lower],
                    'band_upper':[float(v) if np.isfinite(v) else None for v in upper],
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            finally:
                plt.close(fig)
    cache = output / f'trial_{metric}_bins.csv'
    manifest = {'metric':metric, 'units':units, 'x_label':'Normalized Time', 'y_label':'Value',
        'dpi':dpi, 'xlim':[0,1], 'ylim':[0,ymax], 'y_tick_step':ystep,
        'y_ticks':np.arange(0, ymax + 1e-9, ystep).tolist(),
        'baseline_axis_label_pt':BASE_LABEL_PT, 'baseline_tick_label_pt':BASE_TICK_PT,
        'mean_linewidth_pt':MEAN_WIDTH_PT, 'trial_linewidth_pt':TRIAL_WIDTH_PT,
        'spine_linewidth_pt':SPINE_WIDTH_PT, 'labelpad_pt':LABEL_PAD_PT, 'tick_pad_pt':TICK_PAD_PT,
        'trial_quantile_band':{'percentiles':BAND_PERCENTILES, 'color':BAND_COLOR,
            'alpha':BAND_ALPHA, 'zorder':2,
            'aggregation':'Pooled included trials per bin; finite values only; no smoothing'},
        'source_bins':cache.name,
        'source_bins_sha256':hashlib.sha256(cache.read_bytes()).hexdigest() if cache.exists() else None,
        'figures':figures}
    (target / 'plot_style_manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return manifest
