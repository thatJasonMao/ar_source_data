"""Complete and verify a paired V6_2 raw-cell 2D/3D spatial-rule atlas.

Run Plot_Spatial_Rule_3D.py first. This adds only colour bars, colour tables,
the local index, documentation, rerender commands and verification.json.
No training, fitted-model loading or changes to source fields occur. Display
smoothing and nonlinear colour normalization follow the renderer manifest.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import importlib.util
import json
import shutil
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

CHANNELS = ('StaticSign', 'ArrowRibbon', 'MiniMap', 'TextBar')
LABELS = {'StaticSign': 'cividis', 'ArrowRibbon': '162 (rainbow)',
          'MiniMap': '99 (plasma)', 'TextBar': '100 (viridis)'}
EXPECTED_PALETTES = {'StaticSign': None, 'ArrowRibbon': 162, 'MiniMap': 99, 'TextBar': 100}
SUPPLEMENTS = tuple(c + '_colorbar.png' for c in CHANNELS) + (
    'Palette_Overview.png', 'color_scale_summary.csv', 'palette_mapping_256.csv',
    'index.html', 'README.md', 'Rerender.ps1', 'verification.json')
SURFACE_LABELS = {'cells': '原始独立格面', 'bars': '实体三维柱状热力图',
                  'continuous': '连续三维曲面'}


def surface_style(manifest):
    style = manifest.get('surface_style', 'cells')
    require(style in SURFACE_LABELS, 'Unknown 3D surface style: ' + str(style))
    return style


def display_options(manifest):
    saved = manifest.get('display_processing', {})
    return dict(smoothing_iterations=saved.get('smoothing_iterations', 0),
                smoothing_strength=saved.get('smoothing_strength', .5),
                contrast_channels=saved.get('contrast_channels', []),
                contrast_quantile=saved.get('contrast_quantile', .3),
                contrast_position=saved.get('contrast_position', .1),
                contrast_method=saved.get('contrast_method', 'power'))


def verify_smoothing(renderer, part, iterations, strength):
    """Independently repeat valid four-neighbour diffusion and compare heights."""
    raw = part[renderer.FIELD].to_numpy(float)
    actual = renderer.smooth_display_values(part, iterations=iterations, strength=strength)
    keys = np.rint(part[['x', 'z']].to_numpy(float)/renderer.CELL_M-.5).astype(np.int64)
    lookup = {tuple(key): i for i, key in enumerate(keys)}
    neighbors = np.array([[lookup.get((x+dx, z+dz), -1)
                           for dx, dz in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
                          for x, z in keys])
    valid = neighbors >= 0
    counts = valid.sum(1)
    connected = counts > 0
    expected = raw.copy()
    for _ in range(iterations):
        neighbor_sum = np.where(valid, expected[np.maximum(neighbors, 0)], 0).sum(1)
        updated = expected.copy()
        updated[connected] = ((1-strength)*expected[connected]
                              + strength*neighbor_sum[connected]/counts[connected])
        expected = updated
    require(np.allclose(actual, expected, atol=1e-12, rtol=1e-12), 'Display smoothing differs from valid-neighbour diffusion')
    require(np.array_equal(part[renderer.FIELD].to_numpy(float), raw), 'Smoothing changed source values')
    require(actual.min() >= raw.min()-1e-12 and actual.max() <= raw.max()+1e-12,
            'Display smoothing overshoots raw extrema')
    delta = actual-raw
    return actual, dict(iterations=iterations, strength=strength, verified_four_neighbor_diffusion=True,
        raw_minimum=float(raw.min()), raw_maximum=float(raw.max()), display_minimum=float(actual.min()),
        display_maximum=float(actual.max()), max_absolute_change=float(np.abs(delta).max()),
        mean_absolute_change=float(np.abs(delta).mean()), rms_change=float(np.sqrt(np.mean(delta**2))),
        mean_signed_change=float(delta.mean()), isolated_cells=int((~connected).sum()))


def verify_geometry(renderer, part, style, limits, processing=None):
    """Check raw cell ownership, exact centres/tops and the occupied footprint."""
    top, values = renderer.cell_faces(part)
    require(np.array_equal(values, part[renderer.FIELD].to_numpy(float)), 'Raw values differ')
    require(np.array_equal(top[:, :, 2], np.repeat(values[:, None], 4, axis=1)),
            'Raw cell heights differ')
    centers = np.column_stack([-part.x.to_numpy(float), -part.z.to_numpy(float), values])
    require(np.allclose(top[:, :, :2].mean(1), centers[:, :2], atol=1e-12, rtol=0),
            'Display coordinates differ')
    if style == 'cells':
        return dict(cells=len(part), faces=len(top), raw_cell_heights_exact=True)
    if style == 'bars':
        lo, hi = limits
        span = hi - lo if hi > lo else max(1e-6, abs(lo) * .01)
        base = lo - .08 * span
        faces, colors = renderer.bar_faces(part, base)
        require(faces.shape == (6*len(part), 4, 3), 'Each bar needs six quadrilateral faces')
        require(np.array_equal(colors, np.repeat(values, 6)), 'Bar colours change raw utility')
        grouped = faces.reshape(len(part), 6, 4, 3)
        require(np.all(np.isclose(grouped[:, :, :, 2], values[:, None, None], atol=1e-12, rtol=0)
                       | np.isclose(grouped[:, :, :, 2], base, atol=1e-12, rtol=0)),
                'Bar vertices must use the raw top or common display base')
        is_top = np.all(np.isclose(grouped[:, :, :, 2], values[:, None, None], atol=1e-12, rtol=0), axis=2)
        is_bottom = np.all(np.isclose(grouped[:, :, :, 2], base, atol=1e-12, rtol=0), axis=2)
        require(np.all(is_top.sum(1) == 1) and np.all(is_bottom.sum(1) == 1), 'Bar must have one top and one bottom')
        raw_corners = np.rint(top[:, :, :2] / renderer.CELL_M).astype(np.int64)
        actual_corners = np.rint(grouped[:, :, :, :2] / renderer.CELL_M).astype(np.int64)
        require(np.all((actual_corners[:, :, :, None, :] == raw_corners[:, None, None, :, :]).all(-1).any(-1)),
                'Bar extends beyond its original occupied cell')
        return dict(cells=len(part), faces=len(faces), closed_bar_faces_per_cell=6,
                    raw_bar_tops_exact=True, display_base=base)
    processing = processing or display_options({})
    iterations, strength = processing['smoothing_iterations'], processing['smoothing_strength']
    displayed, smoothing = verify_smoothing(renderer, part, iterations, strength)
    centers[:, 2] = displayed
    faces, colors = renderer.continuous_faces(part, smoothing_iterations=iterations, smoothing_strength=strength)
    require(faces.shape == (4*len(part), 3, 3), 'Continuous mesh needs four triangles per occupied cell')
    require(np.array_equal(colors, np.repeat(values, 4)), 'Continuous face colours change raw utility')
    grouped = faces.reshape(len(part), 4, 3, 3)
    require(np.allclose(grouped[:, :, 0, :], centers[:, None, :], atol=1e-12, rtol=0),
            'Continuous mesh does not retain the specified display centres')
    require(np.array_equal(grouped[:, :, 0, 2], np.repeat(displayed[:, None], 4, axis=1)),
            'Continuous centre height differs from the verified display value')
    # Independently compute corner means from verified display heights.
    raw_keys = np.rint(top[:, :, :2] / renderer.CELL_M).astype(np.int64)
    unique, inverse = np.unique(raw_keys.reshape(-1, 2), axis=0, return_inverse=True)
    means = np.bincount(inverse, weights=np.repeat(displayed, 4)) / np.bincount(inverse)
    expected = {tuple(k): float(v) for k, v in zip(unique, means)}
    corners = grouped[:, :, 1:, :]
    corner_keys = np.rint(corners[:, :, :, :2] / renderer.CELL_M).astype(np.int64)
    require(np.all((corner_keys[:, :, :, None, :] == raw_keys[:, None, None, :, :]).all(-1).any(-1)),
            'Continuous triangle bridges an unoccupied cell')
    expected_heights = np.asarray([expected[tuple(k)] for k in corner_keys.reshape(-1, 2)]).reshape(len(part), 4, 2)
    require(np.allclose(corners[:, :, :, 2], expected_heights, atol=1e-12, rtol=1e-12),
            'Shared corner heights differ from local valid-cell means')
    a = grouped[:, :, 1, :2] - grouped[:, :, 0, :2]
    b = grouped[:, :, 2, :2] - grouped[:, :, 0, :2]
    areas = np.abs(a[:, :, 0]*b[:, :, 1] - a[:, :, 1]*b[:, :, 0]) / 2
    require(np.allclose(areas, renderer.CELL_M**2/4, atol=1e-12, rtol=0),
            'Continuous mesh changes the original occupied area')
    return dict(cells=len(part), faces=len(faces), triangles_per_cell=4,
                raw_centers_exact=iterations == 0, display_centers_verified=True, smoothing=smoothing,
                shared_corner_means_verified=True,
                shared_edges_continuous=True, original_cell_footprint_exact=True,
                display_interpolation_only=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_renderer(path):
    spec = importlib.util.spec_from_file_location('atlas_spatial_renderer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def colormap(renderer, channel):
    return (renderer.plt.get_cmap('cividis') if channel == 'StaticSign' else
            renderer.LinearSegmentedColormap.from_list('guide_' + channel,
                                                       renderer.PALETTES[channel], N=256))


def color_ticks(norm, metadata):
    return (np.asarray(norm.inverse(np.linspace(0., 1., 6))) if metadata['kind'] != 'linear'
            else np.linspace(norm.vmin, norm.vmax, 6))


def verify_normalization(renderer, values, channel, manifest, processing):
    _, expected = renderer.make_color_norm(values, channel,
        contrast_channels=processing['contrast_channels'], contrast_quantile=processing['contrast_quantile'],
        contrast_position=processing['contrast_position'], contrast_method=processing['contrast_method'])
    saved = manifest.get('color_normalization', {}).get(channel, expected)
    require(saved == expected, 'Saved colour normalization differs from raw data/parameters: ' + channel)
    norm = renderer.norm_from_metadata(saved)
    minimum, maximum = float(np.min(values)), float(np.max(values))
    require(saved['vmin'] == minimum and saved['vmax'] == maximum, 'Colour endpoints changed')
    require(np.allclose(norm([minimum, maximum]), [0., 1.], atol=1e-12, rtol=0), 'Colour endpoints do not map to 0 and 1')
    samples = np.linspace(0., 1., 1025)
    inverse = np.asarray(norm.inverse(samples), float)
    require(np.all(np.diff(inverse) > 0), 'Inverse colour mapping is not strictly increasing')
    require(np.allclose(norm(inverse), samples, atol=2e-12, rtol=2e-12), 'Colour normalization inverse roundtrip failed')
    if saved['kind'] != 'linear':
        quantile_value = float(np.quantile(values, saved['quantile']))
        require(np.isclose(saved['quantile_value'], quantile_value, atol=1e-12, rtol=1e-12), 'Quantile anchor differs')
        require(np.isclose(float(norm(quantile_value)), saved['target_position'], atol=1e-12, rtol=0),
                'Quantile anchor does not map to target colour position')
    checks = dict(endpoints_preserved=True, strictly_monotonic=True, inverse_roundtrip=True,
                  quantile_anchor_verified=saved['kind'] != 'linear')
    if saved['kind'] == 'piecewise':
        anchor, position = saved['quantile_value'], saved['target_position']
        lower = np.linspace(minimum, anchor, 101)
        upper = np.linspace(anchor, maximum, 101)
        require(np.allclose(norm(lower), np.linspace(0., position, 101), atol=2e-12, rtol=0),
                'Lower interval is not linearly compressed to the requested palette share')
        require(np.allclose(norm(upper), np.linspace(position, 1., 101), atol=2e-12, rtol=0),
                'Values above the quantile are not linearly mapped to the remaining palette')
        checks.update(lower_segment_linear=True, upper_segment_linear=True, no_value_clipping=True)
    return norm, saved, checks


def verify_main_images(renderer, source, output, frame, manifest):
    style = surface_style(manifest)
    processing = display_options(manifest)
    require(manifest.get('status') == 'complete', 'Main figures are not complete')
    require(manifest.get('input_sha256') == renderer.sha256(source), 'Main-figure input hash mismatch')
    require(manifest.get('script_sha256') == renderer.sha256(Path(renderer.__file__)),
            'Main figures were generated by a different renderer; render them again first')
    require(manifest.get('guidances') == list(CHANNELS), 'All four channels are required')
    require(manifest.get('field') == renderer.FIELD, 'Wrong utility field')
    require(manifest.get('cell_m') == 0.4 and manifest.get('dpi') == 300,
            'Atlas requires 0.4 m cells and 300 dpi figures')
    require(manifest.get('rows_rendered') == len(frame), 'Rendered/source row counts differ')
    for flag in ('clipping', 'subsampling', 'raw_data_modified'):
        require(manifest.get(flag) is False, 'Unexpected rendering policy: ' + flag)
    require(manifest.get('smoothing') is (processing['smoothing_iterations'] > 0), 'Incorrect smoothing declaration')
    require(manifest.get('interpolation') is (style == 'continuous'), 'Incorrect interpolation declaration')
    require(tuple(renderer.FIGSIZE) == (16, 13) and renderer.VIEW_ELEVATION == 42
            and renderer.VIEW_AZIMUTH == -65, 'Original figure layout or viewing angle changed')
    expected_layout = dict(figsize_inches=[16, 13], grid=[2, 2], row_order=list(renderer.SMOKES),
        column_order=list(renderer.FLOORS), view_elevation=42, view_azimuth=-65, projection='ortho',
        subplot_adjust=dict(left=.03, right=.86, top=.91, bottom=.09, wspace=.03, hspace=.32),
        colorbar_axes=[.90, .19, .018, .60], title_y=.97, footer_y=.03,
        horizontal_aspect='Physical X/Z ratio', relief_display_aspect=35)
    require(manifest.get('layout') == expected_layout, 'Manifest does not preserve the original 2D layout')
    profile = manifest.get('layout_profile', 'legacy')
    require(profile in ('legacy', 'publication', 'publication-compact'), 'Unknown 3D layout profile')
    layout_3d = manifest.get('layout_3d', manifest.get('layout'))
    expected_3d = renderer.layout_metadata(profile) if hasattr(renderer, 'layout_metadata') else expected_layout
    require(layout_3d == expected_3d, '3D layout does not match the selected renderer profile')
    if profile in ('publication', 'publication-compact'):
        require(layout_3d.get('vertical_utility_axis_visible') is False,
                'Publication layout must hide the vertical utility axis')
        boxes = np.asarray(layout_3d.get('panel_axes'), dtype=float)
        side = 13 * (.91 - .09) / (2 + .32)
        require(boxes.shape == (4, 4), 'Publication layout requires four panel rectangles')
        require(np.allclose(boxes[:, 2:] * [16, 13], side, atol=1e-12, rtol=0),
                'Publication panel dimensions differ from the original active viewports')
        centers = boxes[:, :2] + boxes[:, 2:] / 2
        require(np.allclose(centers[[0, 2], 0], centers[0, 0])
                and np.allclose(centers[[1, 3], 0], centers[1, 0])
                and np.allclose(centers[:2, 1], centers[0, 1])
                and np.allclose(centers[2:, 1], centers[2, 1])
                and np.isclose(centers[0, 0] + centers[1, 0], 1),
                'Publication panels are not aligned and horizontally centered')
        require(layout_3d.get('colorbar_orientation') == 'horizontal'
                and layout_3d.get('footer_y') is None,
                'Publication colorbar or footer policy differs')
        require(layout_3d.get('title_content') == 'Guidance label only'
                and layout_3d.get('title_fontsize') == 36
                and layout_3d.get('panel_title_fontsize') == 26
                and layout_3d.get('title_fontweight') == 'bold'
                and layout_3d.get('panel_title_fontweight') == 'bold',
                'Publication titles do not use the requested text/font policy')
        for key, expected in [('axis_label_fontsize', 11 * 1.08 * 1.5),
                              ('axis_tick_fontsize', 9 * 1.08 * 1.8),
                              ('colorbar_label_fontsize', 11 * 1.08),
                              ('colorbar_tick_fontsize', 11 * 1.08 * 2)]:
            require(np.isclose(layout_3d.get(key, 0), expected),
                    'Publication axis/colorbar font differs from the requested size: ' + key)
    expected_dimensions = (4800, 3900)
    hashes, image_info, summaries, geometry, norms, norm_checks = {}, {}, [], {}, {}, {}
    for channel in tqdm(CHANNELS, desc='Verify paired raw-cell figures'):
        part = frame.loc[frame.guidance.eq(channel)]
        minimum, maximum = float(part[renderer.FIELD].min()), float(part[renderer.FIELD].max())
        require(minimum < maximum, 'Constant field needs a separately documented colour-range policy')
        norm, metadata, norm_checks[channel] = verify_normalization(renderer, part[renderer.FIELD].to_numpy(float), channel, manifest, processing)
        norms[channel] = (norm, metadata)
        records = [manifest[key][channel] for key in ('figures', 'figures_2d')]
        panels = {}
        geometry[channel] = {}
        for smoke in renderer.SMOKES:
            for floor in renderer.FLOORS:
                subset = part.loc[part.smoke.eq(smoke) & part.floor.eq(floor)]
                values = subset[renderer.FIELD].to_numpy(float)
                geometry[channel][smoke + '/' + floor] = verify_geometry(renderer, subset, style, (minimum, maximum), processing)
                panels[smoke + '/' + floor] = dict(cells=len(subset), minimum=float(values.min()),
                                                   maximum=float(values.max()))
        for record, dimension in zip(records, ('3d', '2d')):
            name = channel + '_spatial_demand_' + dimension + '.png'
            require(record.get('file') == name and record.get('dimension') == dimension,
                    'Main figure filename/dimension mismatch')
            require(record.get('palette_number') == EXPECTED_PALETTES[channel]
                    and record.get('palette_reversed') is False, 'Unexpected main-figure palette')
            require(record.get('color_normalization', metadata) == metadata, 'Figure colour normalization differs from channel metadata')
            if profile in ('publication', 'publication-compact') and dimension == '3d':
                actual = record.get('rendered_layout', {})
                actual_boxes = np.asarray(actual.get('panel_axes', []), dtype=float)
                actual_sizes = np.asarray(actual.get('panel_size_inches', []), dtype=float)
                require(actual_boxes.shape == (4, 4)
                        and np.allclose(actual_boxes, layout_3d['panel_axes'], atol=1e-12, rtol=0),
                        'Rendered 3D panel positions differ from the fixed layout: ' + channel)
                require(actual_sizes.shape == (4, 2)
                        and np.allclose(actual_sizes, side, atol=1e-12, rtol=0),
                        'Rendered 3D panel sizes differ from the original viewports: ' + channel)
                require(actual.get('title') == renderer.GUIDANCE_LABELS[channel]
                        and actual.get('title_fontsize') == 36
                        and actual.get('title_fontweight') == 'bold'
                        and actual.get('panel_title_fontsizes') == [26] * 4
                        and actual.get('panel_title_fontweights') == ['bold'] * 4,
                        'Rendered title text/font policy differs: ' + channel)
                axis_fonts = np.asarray(actual.get('axis_label_fontsizes', []), dtype=float)
                require(axis_fonts.shape == (4, 3) and np.allclose(axis_fonts, 11 * 1.08 * 1.5),
                        'Rendered axis label fonts must be 17.82 pt: ' + channel)
                tick_fonts = actual.get('axis_tick_fontsizes', [])
                require(len(tick_fonts) == 4 and all(fonts and np.allclose(fonts, 9 * 1.08 * 1.8)
                                                    for fonts in tick_fonts),
                        'Rendered axis tick fonts must be 17.496 pt: ' + channel)
                cbar_fonts = actual.get('colorbar_tick_fontsizes', [])
                require(cbar_fonts and np.allclose(cbar_fonts, 11 * 1.08 * 2)
                        and np.isclose(actual.get('colorbar_label_fontsize', 0), 11 * 1.08),
                        'Rendered colorbar tick/label fonts differ: ' + channel)
                require(actual.get('colorbar_label_visible') is False,
                        'Publication colorbar annotation must be hidden: ' + channel)
                actual_cbar = np.asarray(actual.get('colorbar_axes', []), dtype=float)
                require(actual.get('colorbar_orientation') == 'horizontal'
                        and actual_cbar.shape == (4,)
                        and np.allclose(actual_cbar, layout_3d['colorbar_axes'], atol=1e-12, rtol=0),
                        'Rendered colorbar is not in the expected bottom horizontal position: ' + channel)
                require(actual.get('panel_background_alpha') == [0, 0, 0, 0],
                        'Adjacent panel backgrounds may hide outward axis labels')
                vertical_axes = actual.get('vertical_utility_axes', [])
                require(len(vertical_axes) == 4 and all(
                    item.get('label') == '' and item.get('ticks') == [] and item.get('minor_ticks') == []
                    and item.get('pane_visible') is True
                    and all(item.get(key) is False for key in ('axis_visible', 'line_visible',
                        'label_visible', 'offset_visible')) for item in vertical_axes),
                    'Vertical utility axis, labels or ticks remain visible: ' + channel)
                require(actual.get('figure_texts') == [renderer.GUIDANCE_LABELS[channel]],
                        'Rendered figure contains an unexpected footer or title: ' + channel)
            for key, value in [('data_minimum', minimum), ('data_maximum', maximum),
                               ('cmap_minimum', minimum), ('cmap_maximum', maximum)]:
                require(np.isclose(record[key], value, atol=1e-10, rtol=1e-10), 'Incorrect ' + key)
            require(set(record['panels']) == set(panels), 'Incomplete panel combinations')
            for key, expected in panels.items():
                require(record['panels'][key]['cells'] == expected['cells'], 'Panel cell count differs')
                for metric in ('minimum', 'maximum'):
                    require(np.isclose(record['panels'][key][metric], expected[metric],
                                       atol=1e-10, rtol=1e-10), 'Panel value range differs')
            with Image.open(output / name) as im:
                dpi = im.info.get('dpi')
                if profile == 'publication-compact' and dimension == '3d':
                    crop = record['rendered_layout']['export_crop']
                    require(crop['original_dimensions'] == list(expected_dimensions)
                            and crop['resampled'] is False and crop['padding_pixels'] == 0,
                            'Unexpected compact crop policy: ' + name)
                    left, top, right, bottom = crop['bbox_pixels']
                    require(0 <= left < right <= expected_dimensions[0]
                            and 0 <= top < bottom <= expected_dimensions[1]
                            and im.size == (right-left, bottom-top) == tuple(crop['dimensions']),
                            'Incorrect compact crop dimensions: ' + name)
                    with Image.open(output / name) as crop_image:
                        pixels = np.asarray(crop_image.convert('RGB'))
                    require(all(np.any(edge != 255) for edge in
                                (pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1])),
                            'Pure-white outer margins remain: ' + name)
                else:
                    require(im.size == expected_dimensions, 'Unexpected main image dimensions: ' + name)
                require(dpi is not None and np.allclose(dpi, (300, 300), atol=0.01, rtol=0),
                        'Unexpected main image DPI: ' + name)
                image_info[name] = dict(dimensions=list(im.size), dpi=list(dpi))
                im.verify()
            hashes[name] = renderer.sha256(output / name)
        require(records[0]['panels'] == records[1]['panels'], '2D and 3D statistics differ')
        summaries.append(dict(guidance=channel, palette=LABELS[channel], reversed=False,
                              vmin=minimum, vmax=maximum, cells=len(part),
                              normalization=metadata['kind'], gamma=metadata['gamma'],
                              quantile=metadata['quantile'], quantile_value=metadata['quantile_value'],
                              target_position=metadata['target_position'],
                              normalization_scope=metadata['normalization_scope']))
    hashes['manifest_3d.json'] = renderer.sha256(output / 'manifest_3d.json')
    return pd.DataFrame(summaries), hashes, image_info, geometry, norms, norm_checks


def palette_table(renderer, scales, reference, norms):
    old = pd.read_csv(reference / 'palette_mapping_256.csv', float_precision='round_trip')
    rgba_columns = ['red', 'green', 'blue', 'alpha']
    records, checks = [], {}
    normalized = np.linspace(0., 1., 256)
    from matplotlib.colors import to_hex
    for scale in scales.itertuples(index=False):
        channel = scale.guidance
        norm, metadata = norms[channel]
        cmap = colormap(renderer, channel)
        colors = np.asarray(cmap(normalized))
        prior = old.loc[old.guidance.eq(channel)].sort_values('index')
        require(len(prior) == 256 and np.array_equal(prior['index'].to_numpy(), np.arange(256)),
                'Reference must contain exactly 256 indexed colours: ' + channel)
        difference = float(np.max(np.abs(colors - prior[rgba_columns].to_numpy(float))))
        require(np.allclose(colors, prior[rgba_columns].to_numpy(float), rtol=0, atol=1e-15),
                'Renderer RGBA differs from the reference palette: ' + channel)
        require(prior.palette.eq(LABELS[channel]).all(), 'Reference palette label differs')
        for index, (sample, rgba) in enumerate(zip(normalized, colors)):
            hex_color = to_hex(rgba, keep_alpha=False)
            require(hex_color.lower() == str(prior.iloc[index]['hex']).lower(), 'Reference HEX differs')
            records.append(dict(guidance=channel, palette=LABELS[channel], index=index,
                normalized_sample=float(sample), utility_sample=float(norm.inverse(sample)),
                utility_bin_low=float(norm.inverse(index / 256)),
                utility_bin_high=float(norm.inverse((index+1) / 256)),
                normalization=metadata['kind'], gamma=metadata['gamma'],
                red=float(rgba[0]), green=float(rgba[1]), blue=float(rgba[2]), alpha=float(rgba[3]), hex=hex_color))
        checks[channel] = dict(colors=256, reference_rgba_equal=True, rgba_max_absolute_difference=difference,
                               reference_hex_equal=True, reversed=False)
    return pd.DataFrame(records), checks


def draw_colorbars(renderer, scales, output, norms, hide_annotation=False):
    plt = renderer.plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 11})
    from matplotlib import font_manager
    font = font_manager.findfont(font_manager.FontProperties(family='Times New Roman'), fallback_to_default=False)
    for row in tqdm(scales.itertuples(index=False), total=len(scales), desc='Render standalone colour bars'):
        fig = plt.figure(figsize=(8, 1.5))
        try:
            ax = fig.add_axes([.08, .40, .84, .25])
            norm, metadata = norms[row.guidance]
            bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap(renderer, row.guidance)),
                               cax=ax, orientation='horizontal')
            bar.set_ticks(color_ticks(norm, metadata))
            ax.set_title(row.guidance + ' | ' + row.palette, loc='left', fontsize=13, pad=8)
            if hide_annotation:
                bar.ax.xaxis.label.set_visible(False)
            else:
                bar.set_label('Relative information utility (signed reward contrast)', fontsize=10)
            fig.savefig(output / (row.guidance + '_colorbar.png'), dpi=300, facecolor='white')
        finally:
            plt.close(fig)
    fig = plt.figure(figsize=(10, 6.5))
    try:
        fig.suptitle('V6_2 | Palette and raw utility ranges', fontsize=18, y=.965)
        for index, row in enumerate(scales.itertuples(index=False)):
            norm, metadata = norms[row.guidance]
            ax = fig.add_axes([.10, .79 - .20*index, .80, .045])
            bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm,
                               cmap=colormap(renderer, row.guidance)), cax=ax, orientation='horizontal')
            bar.set_ticks(color_ticks(norm, metadata))
            bar.ax.xaxis.label.set_visible(False)
            ax.set_title(row.guidance + ' | ' + row.palette, loc='left', fontsize=13, pad=8)
        fig.text(.5, .035, 'Same mapping in 2D and 3D. Each channel has its own numerical range.',
                 ha='center', fontsize=12)
        fig.savefig(output / 'Palette_Overview.png', dpi=300, facecolor='white')
    finally:
        plt.close(fig)
    return font


def visibility_metadata(source):
    columns = pd.read_csv(source, nrows=0).columns
    if 'visibility_m' not in columns:
        return None
    frame = pd.read_csv(source, usecols=['smoke', 'visibility_m'])
    result = {}
    for smoke, part in frame.groupby('smoke'):
        values = part.visibility_m.unique()
        require(len(values) == 1 and np.isfinite(values[0]) and values[0] > 0,
                'Ambiguous source visibility')
        result[str(smoke)] = float(values[0])
    return result


def write_documents(source, renderer_path, reference, output, scales, visibility, style, processing, norms,
                    layout_profile='legacy', basemap=None):
    e = html.escape
    iterations, strength = processing['smoothing_iterations'], processing['smoothing_strength']
    smoothing_text = (f'显示高度进行{iterations}轮四邻接平滑，强度{strength:g}：每轮新值=(1-强度)×原显示值+强度×有效四邻格均值。'
        '仅沿共享边的有效格子传播，孤立格保持原值，不跨缺格、墙洞或空白区域，不产生超出原值域的高度。'
        '这会改变部分格点中心的显示高度，原始CSV及用于颜色的原始值保持不变。' if iterations else
        '未启用显示高度平滑，格点中心仍保持原始高度。')
    geometry_description = {
        'cells': '原始独立格面：每个0.4 m格子保持原始平顶高度，无插值。',
        'bars': '实体三维柱状图：每个0.4 m格子生成闭合六面柱，柱顶精确等于原始即时奖励差。柱底沿用原图显示基面（该通道最小值减0.08倍值域），不是零值；柱长不是奖励绝对值。',
        'continuous': '连续三维曲面：'+smoothing_text+'共享角点高度取接触该角的有效格点显示高度均值，每格中心连接四角形成四个三角形。共享边连续，只在原有效格子内部进行分片线性显示插值，不跨越建筑空缺区域。',
    }[style]
    intro = ('4 通道 × 2D/3D，共 8 张 300 dpi 主图。每张图含高低烟雾与站厅/站台。'
             '保留原始0.4 m格子足迹和色标原始极值端点，不做KDE、全域补洞、抽样或截断；3D显示奖励差的空间形状，不是建筑高程。'
             '数值不是概率或因果效应。同一通道全部面板共用色标，跨通道颜色不能直接比较。')
    layout_text = (
        '3D采用最终紧凑布局：大标题只保留引导范式名，36 pt加粗；子图标题26 pt加粗并收紧间距；'
        '坐标轴刻度数字17.496 pt（在上一版9.72 pt基础上增加80%），轴名称17.82 pt（11.88 pt再增加50%），'
        '色条刻度数字23.76 pt（11.88 pt再增加100%）；最终图中取消色条底部说明文字。'
        '站厅X轴名称间距为14 pt。隐藏垂直数值轴线、轴名称、刻度和刻度文字；保留水平X/Z坐标轴。'
        '曲面高度、颜色、坐标范围及底部水平色条不变，无底部批注。'
        '大图保持16 × 13英寸、300 dpi、4800 × 3900像素；四个3D绘图区保持原有约4.5948 × 4.5948英寸的相同尺寸，'
        '以统一投影缩放减少内部留白，横纵位置和标题间距统一收紧，未裁切或缩放导出的图片。'
        '2D图及独立色条沿用原布局。详细位置、投影缩放和字体记录于manifest的layout_3d。'
        if layout_profile in ('publication', 'publication-compact') else
        '2D/3D沿用原布局及色条位置。')
    if layout_profile == 'publication-compact':
        layout_text = layout_text.replace(
            '大图保持16 × 13英寸、300 dpi、4800 × 3900像素；',
            '内部画布16 × 13英寸、300 dpi；导出时裁掉四周全部纯白边缘，最终尺寸见manifest；')
        layout_text = layout_text.replace(
            '以统一投影缩放减少内部留白，横纵位置和标题间距统一收紧，未裁切或缩放导出的图片。',
            '子图标题下移5 pt，下排子图及色条上移0.0975英寸（300 dpi时29.25像素）；仅裁外围空白，不重采样。')
    if basemap and basemap.get('enabled'):
        layout_text += (f' 3D实况底图版本：{basemap.get("variant", "walls")}；'
                        f'站厅使用{basemap["mapping"]["Hall"]}、站台使用{basemap["mapping"]["Platform"]}；'
                        '按Unity正交相机逆投影到真实地坪（Hall Y=-11.8 m、Platform Y=-15.771 m），'
                        '再以当前倾斜视角仿射绘制完整分辨率底图，透明度0.74，坐标为-Unity X/Z。'
                        '底图位于曲面下方，按现有水平轴范围裁剪。墙和设施保留原截图的立体细节，未重建建筑几何。')
    sections = []
    for row in scales.itertuples(index=False):
        pairs = ''.join(f'<a href="{row.guidance}_spatial_demand_{d}.png"><img loading="lazy" '
                        f'src="{row.guidance}_spatial_demand_{d}.png" alt="{row.guidance} {d}"></a>' for d in ('2d', '3d'))
        sections.append(f'<section><h2>{row.guidance}<small>{e(row.palette)} · {row.vmin:.6f} 至 {row.vmax:.6f}</small>'
                        f'</h2><div class="pair">{pairs}</div></section>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V6_2 2D / 3D 热力图</title><style>body{font:16px/1.7 system-ui,sans-serif;background:#f4f5f7;color:#20242a;max-width:1500px;margin:auto;padding:28px}h1{margin-bottom:8px}h2{font-size:23px}small{font-size:14px;font-weight:400;margin-left:16px}section{background:white;padding:20px;margin:24px 0;border-radius:12px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}img{width:100%;height:auto}a{color:#2759a0}.legend{max-width:900px}p{max-width:1100px}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>
<h1>V6_2 · 2D / 3D 热力图</h1>'''
    page += '<p><strong>' + SURFACE_LABELS[style] + '</strong>：' + geometry_description + '</p>'
    q, position = processing['contrast_quantile'], processing['contrast_position']
    selected_channels = [channel for channel in CHANNELS if norms[channel][1]['kind'] != 'linear']
    if not selected_channels:
        transform_text = '全部通道使用线性映射。'
    elif processing['contrast_method'] == 'piecewise':
        transform_text = ('、'.join(selected_channels) + '采用两段连续线性映射：'
            f'原始最小值至P{100*q:g}使用色带前{100*position:g}%，'
            f'P{100*q:g}至原始最大值线性使用剩余{100*(1-position):g}%色带。'
            f'仅部分压缩最低{100*q:g}%数据所在的区间，中低值、中高值和高值区间不再经过全局幂函数压缩。'
            '其他通道维持原线性映射。')
    else:
        transform_text = ('、'.join(selected_channels) + '使用PowerNorm幂函数映射：'
            f'P{100*q:g}对应色带{100*position:g}%位置；其他通道维持线性映射。')
    mapping_text = ('颜色始终使用原始格点值，色标最小值和最大值不变。' + transform_text
        + '所有面板、二维图和独立色条使用同一通道映射；没有删除、裁掉或合并极低值。')
    page += '<p>' + mapping_text + '</p><p>' + layout_text + '</p>'
    page += '<p>' + intro + '</p><p>视距：' + e(str(visibility)) + '</p>'
    page += '<p><a href="README.md">方法说明</a> · <a href="color_scale_summary.csv">色标范围</a> · '
    page += '<a href="palette_mapping_256.csv">256 级颜色数值映射</a> · <a href="verification.json">核验记录</a></p>'
    page += '<section class="legend"><img src="Palette_Overview.png" alt="调色板与数值范围"></section>'
    page += ''.join(sections) + '</html>\n'
    (output/'index.html').write_text(page, encoding='utf-8')
    table = '\n'.join(f'|{r.guidance}|{r.palette}|{r.vmin:.9f}|{r.vmax:.9f}|{r.normalization}|'
        + (f'{r.gamma:.8g}|' if pd.notna(r.gamma) else '—|')
        + (f'{r.quantile_value:.9f} → {r.target_position:g}|' if pd.notna(r.quantile_value) else '—|')
        for r in scales.itertuples(index=False))
    readme = f'''# V6_2 {SURFACE_LABELS[style]}与配套2D图集

{intro}

四通道各一张2D和一张3D主图，全部300 dpi；2D为4800 × 3900像素，3D尺寸以manifest的导出记录为准；坐标为 -Unity X、-Unity Z。视角（elev=42°, azim=-65°）、正交投影、两烟雾×两楼层四面板顺序保持一致。

{layout_text}

模式：`{style}`。{geometry_description}

{mapping_text}

颜色由所属原始格子的 `relative_information_utility` 经记录的线性、幂函数或分段线性归一化映射，调色板256级RGBA和方向不变。所有柱面或该格四个三角形使用原始格值颜色，不以平滑后的高度、柱长或插值角点重新计算颜色，也不加光照。2D仅改变选定通道的颜色映射，不进行空间平滑。3D显示高度按上文参数处理；不存在原始数据改写或裁掉离群值。

令 `u=(value-vmin)/(vmax-vmin)`。`linear`映射为`t=u`；历史`power`映射为`t=u**gamma`，其中`gamma=log(target_position)/log((Pq-vmin)/(vmax-vmin))`。`piecewise`令`a=Pq`、`p=target_position`：当`value<=a`时，`t=p*(value-vmin)/(a-vmin)`；当`value>a`时，`t=p+(1-p)*(value-a)/(vmax-a)`。两段在Pq处连续，分别保持线性，原最小值/最大值仍映射0/1。分段映射不使用gamma，表中留空。每通道的Pq基于两烟雾×两楼层全部原始有效格子，不按面板单独调整。色条在等间距颜色位置显示逆变换得到的原数值刻度。

每个通道的色标使用**本次输入空间场**两烟雾、两楼层全部数值的最小值/最大值。保留旧图集的调色板及方向，但不复用旧模型的数值范围。不同通道是独立拟合的模型，色标也独立，不能根据颜色直接比较跨通道需求大小。颜色和高度均不是概率、因果效应或物理高程。

输入视距：`{visibility}`。

|通道|调色板（未反转）|最小值|最大值|映射|gamma|分位数值 → 色带位置|
|---|---|---:|---:|---|---:|---|
{table}

文件说明：
- `index.html`：本地浏览图集，点击图片查看原图。
- `*_spatial_demand_2d.png` / `*_spatial_demand_3d.png`：8张主图，每张包含四面板。
- `Palette_Overview.png` / `*_colorbar.png`：色标总览与四条独立色条，均300 dpi。
- `color_scale_summary.csv`：每通道原始极值、映射种类、分位数锚点和目标色带位置；gamma只用于幂函数映射，分段映射留空。
- `palette_mapping_256.csv`：256级RGBA、HEX、代表值和覆盖区间；数值列由该通道映射的逆函数计算。区间左闭右开，最后一级包含最大值；索引`min(floor(256*norm(value)),255)`。浮点RGBA列保存精确颜色。
- `manifest_3d.json`：原绘图脚本生成的逐面板格数、极值、绘图方法和哈希记录。
- `verification.json`：核验原始值、显示平滑/共享边、每面板原始高度与显示高度偏差（最大、平均绝对值、RMS）、颜色端点/单调性/逆变换/分位锚点、图片尺寸和256级RGBA；不修改主图manifest。
- `Rerender.ps1`：先调用原主绘图脚本，再调用本补全脚本，重现完整20文件图集。默认输出到新的`Rerender`目录，拒绝覆盖已存在目录。

父目录原来的 `*_spatial_demand.png` 是训练脚本生成的显示平滑图；本目录按上述参数改变3D显示几何和选定通道颜色映射。原始空间场、模型、训练过程、其他图集和缓存均未更改。

原场：`{source}`

主图脚本：`{renderer_path}`

补全脚本：`{Path(__file__).resolve()}`

调色板与图集结构参考：`{reference}`
'''
    (output/'README.md').write_text(readme, encoding='utf-8')
    def psquote(value):
        return "'" + str(value).replace("'", "''") + "'"
    extra = (f' --smooth-iterations {iterations} --smooth-strength {strength:g}'
             f' --contrast-quantile {processing["contrast_quantile"]:g} --contrast-position {processing["contrast_position"]:g}'
             f' --contrast-method {psquote(processing["contrast_method"])}')
    if layout_profile != 'legacy':
        extra += ' --layout-profile ' + psquote(layout_profile)
    if processing['contrast_channels']:
        extra += ' --contrast-channels ' + psquote(','.join(processing['contrast_channels']))
    if basemap and basemap.get('enabled'):
        extra += ' --basemap-root ' + psquote(basemap['root'])
        extra += ' --basemap-variant ' + psquote(basemap.get('variant', 'walls'))
    script = f'''param([string]$Output = (Join-Path $PSScriptRoot 'Rerender'), [string]$Python = 'python')
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $Output) {{ throw "Output already exists: $Output" }}
& $Python -B {psquote(renderer_path)} --input {psquote(source)} --output $Output --dpi 300 --surface-style {psquote(style)}{extra}
if ($LASTEXITCODE -ne 0) {{ throw "Main plotting failed: $LASTEXITCODE" }}
& $Python -B {psquote(Path(__file__).resolve())} --input {psquote(source)} --output $Output --reference {psquote(reference)} --renderer {psquote(renderer_path)}
if ($LASTEXITCODE -ne 0) {{ throw "Atlas completion failed: $LASTEXITCODE" }}
'''
    (output/'Rerender.ps1').write_text(script, encoding='utf-8-sig')


def complete(source, output, reference, renderer_path, force=False):
    source, output, reference, renderer_path = (Path(p).resolve() for p in (source, output, reference, renderer_path))
    require(output.is_dir(), 'Render the main images before completing the atlas')
    require(output != reference, 'Never overwrite the reference atlas')
    require(source.is_file() and renderer_path.is_file(), 'Input or renderer is missing')
    for name in ('manifest_3d.json', 'palette_mapping_256.csv'):
        require((reference/name).is_file(), 'Reference atlas is incomplete: ' + name)
    conflicts = [name for name in SUPPLEMENTS if (output/name).exists()]
    require(force or not conflicts, 'Supplemental files already exist; use --force only to replace these known files: ' + str(conflicts))
    renderer = load_renderer(renderer_path)
    source_hash = renderer.sha256(source)
    reference_hashes = {name: renderer.sha256(reference/name) for name in ('manifest_3d.json', 'palette_mapping_256.csv')}
    manifest = read_json(output/'manifest_3d.json')
    old_manifest = read_json(reference/'manifest_3d.json')
    style = surface_style(manifest)
    processing = display_options(manifest)
    metadata_reader = getattr(renderer, 'basemap_metadata', lambda root: dict(enabled=False))
    basemap = manifest.get('basemap', json.loads(json.dumps(metadata_reader(None))))
    metadata_options = dict(variant=basemap['variant']) if 'variant' in basemap else {}
    expected_basemap = json.loads(json.dumps(metadata_reader(
        basemap.get('root') if basemap.get('enabled') else None, **metadata_options)))
    if 'variant' not in basemap:
        expected_basemap.pop('variant', None)
    require(basemap == expected_basemap, 'Basemap metadata does not match the installed source images')
    require(all(manifest.get('figures', {}).get(channel, {}).get('basemap_enabled', False) == bool(basemap.get('enabled'))
                for channel in CHANNELS), '3D figure basemap policy differs from manifest')
    require(all(manifest.get('figures_2d', {}).get(channel, {}).get('basemap_enabled', False) is False
                for channel in CHANNELS), 'Basemap must not alter paired 2D figures')
    basemap_checks = {}
    if basemap.get('enabled'):
        # Forward-project independent world points with the documented Unity
        # camera, then check the renderer's screenshot inverse, including signs.
        yaw, pitch = np.deg2rad([225., 35.264])
        right = np.asarray([np.cos(yaw), 0., -np.sin(yaw)])
        up = np.asarray([np.sin(yaw)*np.sin(pitch), np.cos(pitch), np.cos(yaw)*np.sin(pitch)])
        for floor, floor_y in [('Hall', -11.8), ('Platform', -15.771)]:
            world = np.asarray([[0., floor_y, 0.], [4., floor_y, -30.],
                                [-4., floor_y, 30.], [0., floor_y, -60.]])
            relative = world-np.asarray([81., 71.2, 69.1])
            screen_x = ((relative @ right)/(53.*1920./1080.)+1.)*960.
            screen_y = (1.-(relative @ up)/53.)*540.
            width, height = basemap['files'][floor]['size']
            fitted_width = round(width*1080./height)
            u = (screen_x-(1920.-fitted_width)/2.)*width/fitted_width
            v = screen_y*height/1080.
            actual = renderer.basemap_world_xz(u, v, floor, (width, height))
            error = float(np.max(np.abs(actual-world[:, [0, 2]])))
            require(error < 1e-10, 'Basemap world/screenshot registration roundtrip failed: '+floor)
            basemap_checks[floor] = dict(floor_y=floor_y, world_roundtrip_max_error_m=error,
                positive_and_negative_xz_verified=True, screenshot_file=basemap['files'][floor]['filename'])
    smoothed = processing['smoothing_iterations'] > 0
    frame = renderer.load_data(source, CHANNELS)
    scales, main_hashes, main_image_info, geometry, norms, norm_checks = verify_main_images(renderer, source, output, frame, manifest)
    mapping, palette_checks = palette_table(renderer, scales, reference, norms)
    policies = ('field', 'cell_m', 'clipping', 'subsampling',
                'display_coordinates', 'cmap_scope', 'missing_cell_policy')
    require(all(manifest.get(k) == old_manifest.get(k) for k in policies), 'Shared data/colour policy differs from reference')
    changed_policies = {k: dict(reference=old_manifest.get(k, 'cells' if k == 'surface_style' else None),
                               current=manifest.get(k))
                        for k in ('surface_style', 'geometry_policy', 'smoothing', 'interpolation', 'height',
                                  'paired_2d_policy', 'display_processing', 'color_normalization', 'basemap')
                        if old_manifest.get(k, 'cells' if k == 'surface_style' else None) != manifest.get(k)}
    profile = manifest.get('layout_profile', 'legacy')
    layout_3d = manifest.get('layout_3d', manifest.get('layout'))
    reference_layout_3d = old_manifest.get('layout_3d', old_manifest.get('layout'))
    for key, old_value, value in (
            ('layout_profile', old_manifest.get('layout_profile', 'legacy'), profile),
            ('layout_3d', reference_layout_3d, layout_3d)):
        if old_value != value:
            changed_policies[key] = dict(reference=old_value, current=value)
    same_layout = (manifest.get('layout') == old_manifest.get('layout')
                   and layout_3d == reference_layout_3d)
    same_input = old_manifest.get('input_sha256') == source_hash
    if old_manifest.get('layout'):
        require(manifest.get('layout') == old_manifest['layout'], 'Reference 2D layout differs')
    preserved_reference_files = {}
    unchanged_mapping_channels = []
    if same_input:
        for channel in CHANNELS:
            current = norms[channel][1]
            old = old_manifest.get('color_normalization', {}).get(channel, {})
            old_figure = old_manifest['figures'][channel]
            equivalent = (old.get('kind', 'linear') == current['kind']
                and old.get('gamma', 1.) == current['gamma']
                and old.get('vmin', old_figure['cmap_minimum']) == current['vmin']
                and old.get('vmax', old_figure['cmap_maximum']) == current['vmax']
                and old.get('quantile') == current['quantile']
                and old.get('quantile_value') == current['quantile_value']
                and old.get('target_position') == current['target_position'])
            if not equivalent:
                continue
            unchanged_mapping_channels.append(channel)
            name = channel + '_spatial_demand_2d.png'
            require(renderer.sha256(output/name) == renderer.sha256(reference/name), '2D image changed: ' + name)
            preserved_reference_files[name] = renderer.sha256(reference/name)
    visibility = visibility_metadata(source)
    exported_files = {source.name: source_hash}
    export_manifest = source.parent/'manifest.json'
    if export_manifest.exists():
        record = read_json(export_manifest)
        for name, expected in record.get('files_sha256', {}).items():
            path = (source.parent/name).resolve()
            require(path.is_relative_to(source.parent) and path.is_file(), 'Invalid export-file path')
            require(renderer.sha256(path) == expected, 'Parent export hash mismatch: ' + name)
            exported_files[name] = expected
        exported_files['manifest.json'] = renderer.sha256(export_manifest)
    with tempfile.TemporaryDirectory(prefix='.atlas-', dir=output) as temporary:
        stage = Path(temporary).resolve()
        require(stage.is_relative_to(output), 'Temporary output outside target directory')
        font = draw_colorbars(renderer, scales, stage, norms, hide_annotation=profile in ('publication', 'publication-compact'))
        scales.to_csv(stage/'color_scale_summary.csv', index=False)
        mapping.to_csv(stage/'palette_mapping_256.csv', index=False)
        write_documents(source, renderer_path, reference, stage, scales, visibility, style, processing, norms,
                        profile, basemap)
        if same_input:
            names = ([c+'_colorbar.png' for c in unchanged_mapping_channels]
                     if profile not in ('publication', 'publication-compact') else [])
            if len(unchanged_mapping_channels) == len(CHANNELS):
                names.append('Palette_Overview.png')
            for name in names:
                require(renderer.sha256(stage/name) == renderer.sha256(reference/name),
                        'Existing colour mapping artifact changed: ' + name)
                preserved_reference_files[name] = renderer.sha256(reference/name)
        supplemental_info = {}
        for name, expected_size in [(c+'_colorbar.png', (2400, 450)) for c in CHANNELS] + [('Palette_Overview.png', (3000, 1950))]:
            with Image.open(stage/name) as im:
                dpi = im.info.get('dpi')
                require(im.size == expected_size and dpi and np.allclose(dpi, (300, 300), atol=.01, rtol=0),
                        'Incorrect supplemental image dimensions/DPI')
                supplemental_info[name] = dict(dimensions=list(im.size), dpi=list(dpi))
                im.verify()
        require(all(renderer.sha256(output/name) == value for name, value in main_hashes.items()),
                'Main images or manifest changed while completing the atlas')
        require(all(renderer.sha256(source.parent/name) == value for name, value in exported_files.items()),
                'Original export files changed')
        require(all(renderer.sha256(reference/name) == value for name, value in reference_hashes.items()),
                'Reference palette/manifest changed')
        hashes = dict(main_hashes)
        hashes.update({name: renderer.sha256(stage/name) for name in SUPPLEMENTS if name != 'verification.json'})
        verification = dict(status='passed', created_at_utc=datetime.now(timezone.utc).isoformat(),
            rows=len(frame), channels=4, panels_per_dimension=16, source_path=str(source), source_sha256=source_hash,
            smoke_visibility_m=visibility, original_export_files_unchanged=True, original_export_files_sha256=exported_files,
            surface_style=style, geometry_policy=manifest.get('geometry_policy'), geometry_checks=geometry,
            display_processing=processing, color_normalization={c: norms[c][1] for c in CHANNELS},
            normalization_checks=norm_checks, unchanged_mapping_channels=unchanged_mapping_channels,
            same_rendering_policy_as_reference=not changed_policies, changed_rendering_policies=changed_policies,
            same_layout_as_reference=same_layout, layout=manifest.get('layout'),
            layout_profile=profile, layout_3d=layout_3d,
            basemap=basemap,
            basemap_registration_checks=basemap_checks,
            same_palette_provenance_as_reference=(manifest['palette_provenance'] == old_manifest['palette_provenance']),
            reference_has_same_input=same_input, reference_artifacts_byte_identical=preserved_reference_files,
            renderer_path=str(renderer_path), renderer_sha256=renderer.sha256(renderer_path),
            reference_renderer_sha256=old_manifest['script_sha256'], reference_path=str(reference), reference_files_sha256=reference_hashes,
            atlas_completion_script=str(Path(__file__).resolve()), atlas_completion_script_sha256=renderer.sha256(Path(__file__)),
            paired_2d_3d_scales_and_statistics_equal=True, raw_cell_heights_exact=style != 'continuous',
            raw_centers_exact=not smoothed, display_centers_verified=True,
            raw_bar_tops_exact=style == 'bars', original_occupied_footprint_exact=True,
            palette_256_rgba_equal_to_reference=True, palette_checks=palette_checks,
            image_dimensions=(None if profile == 'publication-compact' else [4800, 3900]), dpi=300, main_image_info=main_image_info, supplemental_image_info=supplemental_info,
            verified_main_images=[name for name in main_hashes if name.endswith('.png')],
            font_path=font, smoothing=smoothed, interpolation=style == 'continuous', clipping=False, subsampling=False,
            training_executed=False, caches_used=False, files_sha256=hashes,
            hash_scope='All 19 deliverables other than this verification.json; unknown pre-existing files are untouched.')
        (stage/'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding='utf-8')
        for name in SUPPLEMENTS:
            # Copy into ordinary output files so a Windows private temporary-directory
            # ACL is not inherited by the published user-facing artifacts.
            shutil.copyfile(stage/name, output/name)
    require(all(renderer.sha256(output/name) == value for name, value in hashes.items()), 'Published atlas hash mismatch')
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, dest='source')
    parser.add_argument('--output', type=Path, required=True, help='Directory already containing eight main images and manifest_3d.json')
    parser.add_argument('--reference', type=Path, required=True, help='Historical Results_3D atlas whose palette is retained')
    parser.add_argument('--renderer', type=Path, default=Path(__file__).resolve().with_name('Plot_Spatial_Rule_3D.py'))
    parser.add_argument('--force', action='store_true', help='Replace only this script\'s eleven named supplemental files')
    args = parser.parse_args()
    result = complete(args.source, args.output, args.reference, args.renderer, args.force)
    print(json.dumps({k: result[k] for k in ('status', 'rows', 'source_sha256', 'smoke_visibility_m',
        'palette_256_rgba_equal_to_reference', 'paired_2d_3d_scales_and_statistics_equal', 'training_executed')}, indent=2))
    print('ATLAS=' + str(args.output.resolve()))


if __name__ == '__main__':
    main()
