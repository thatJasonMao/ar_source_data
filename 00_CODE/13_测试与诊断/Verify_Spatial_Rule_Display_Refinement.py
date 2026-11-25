"""Verify display smoothing and quantile-anchored colour normalization; no training.

Optionally verify the selected channels against a real exported field with --input.
This script reads inputs and reports checks without changing data or figures.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PLOT = None
SELECTED = ('StaticSign', 'TextBar')


def cells(positions, values):
    return pd.DataFrame({'x': [(i+.5)*.4 for i, j in positions],
                         'z': [(j+.5)*.4 for i, j in positions],
                         'relative_information_utility': values})


class DisplayRefinement(unittest.TestCase):
    def test_diffusion_uses_only_shared_edges_and_never_bridges_a_gap(self):
        # Two two-cell components, plus a cell touching only diagonal corners.
        part = cells([(0, 0), (1, 0), (3, 0), (4, 0), (2, 1)],
                     [0., 20., 100., 140., -999.])
        before = part.copy(deep=True)
        expected = [10., 10., 120., 120., -999.]
        for iterations in (1, 4, 12):
            actual = PLOT.smooth_display_values(part, iterations=iterations, strength=.5)
            np.testing.assert_array_equal(actual, expected)
        pd.testing.assert_frame_equal(part, before)

    def test_smoothing_preserves_constants_and_bounds_and_reduces_a_spike(self):
        positions = [(i, j) for i in range(3) for j in range(3)]
        flat = cells(positions, [-2.75]*9)
        np.testing.assert_array_equal(PLOT.smooth_display_values(flat, iterations=4, strength=.5),
                                      flat[PLOT.FIELD].to_numpy())
        spike = cells(positions, [0., 0., 0., 0., 100., 0., 0., 0., 0.])
        before = spike.copy(deep=True)
        raw = spike[PLOT.FIELD].to_numpy()
        np.testing.assert_array_equal(PLOT.smooth_display_values(spike), raw)
        once = PLOT.smooth_display_values(spike, iterations=1, strength=.5)
        final = PLOT.smooth_display_values(spike, iterations=4, strength=.5)
        self.assertEqual(once[4], 50.)
        self.assertLess(final.max(), once.max())
        self.assertGreaterEqual(final.min(), raw.min())
        self.assertLessEqual(final.max(), raw.max())
        pd.testing.assert_frame_equal(spike, before)

    def test_smoothed_mesh_keeps_raw_colours_and_observed_footprint(self):
        positions = [(i, j) for i in range(3) for j in range(3) if (i, j) != (1, 1)]
        part = cells(positions, [-100., 0., 80., -30., 5., 100., -2., 50.])
        before = part.copy(deep=True)
        display = PLOT.smooth_display_values(part, iterations=4, strength=.5)
        faces, colours = PLOT.continuous_faces(part, smoothing_iterations=4, smoothing_strength=.5)
        self.assertEqual(faces.shape, (4*len(part), 3, 3))
        np.testing.assert_array_equal(colours, np.repeat(part[PLOT.FIELD].to_numpy(), 4))
        self.assertFalse(np.array_equal(display, part[PLOT.FIELD].to_numpy()))
        group = faces.reshape(len(part), 4, 3, 3)
        np.testing.assert_array_equal(group[:, :, 0, 2], np.repeat(display[:, None], 4, axis=1))
        centers = -part[['x', 'z']].to_numpy()
        self.assertTrue(np.all(np.abs(group[:, :, :, :2]-centers[:, None, None, :]) <= .2+1e-12))
        a = group[:, :, 1, :2]-group[:, :, 0, :2]
        b = group[:, :, 2, :2]-group[:, :, 0, :2]
        areas = np.abs(a[:, :, 0]*b[:, :, 1]-a[:, :, 1]*b[:, :, 0])/2
        np.testing.assert_allclose(areas.sum(axis=1), np.full(len(part), .16), rtol=0, atol=1e-12)
        vertices = {}
        for x, y, z in faces.reshape(-1, 3):
            key = (round(x, 10), round(y, 10))
            if key in vertices:
                self.assertAlmostEqual(z, vertices[key], places=12)
            vertices[key] = z
        pd.testing.assert_frame_equal(part, before)

    def test_power_mapping_anchors_inverse_and_palette_endpoints(self):
        values = np.array([-8., -4., -3., -2., -1.5, -1.2, -1., -.7, -.3, .25])
        raw_copy = values.copy()
        for channel in SELECTED:
            with self.subTest(channel=channel):
                norm, metadata = PLOT.make_color_norm(values, channel, contrast_channels=SELECTED,
                    contrast_quantile=.3, contrast_position=.1)
                q30 = np.quantile(values, .3)
                self.assertAlmostEqual(float(norm(q30)), .1, places=12)
                np.testing.assert_allclose(norm([values.min(), values.max()]), [0., 1.], rtol=0, atol=1e-14)
                grid = np.linspace(values.min(), values.max(), 101)
                self.assertTrue(np.all(np.diff(norm(grid)) > 0))
                np.testing.assert_allclose(norm.inverse(norm(grid)), grid, rtol=0, atol=1e-12)
                reconstructed = PLOT.norm_from_metadata(metadata)
                np.testing.assert_allclose(reconstructed(grid), norm(grid), rtol=0, atol=1e-14)
                cmap = (PLOT.plt.get_cmap('cividis') if channel == 'StaticSign' else
                        PLOT.LinearSegmentedColormap.from_list('TextBar_test', PLOT.PALETTES[channel], N=256))
                np.testing.assert_array_equal(cmap(norm([values.min(), values.max()])), cmap([0., 1.]))
        np.testing.assert_array_equal(values, raw_copy)

    def test_unselected_channels_and_default_mapping_remain_linear(self):
        values = np.array([-8., -3., -1., .25])
        expected = (values-values.min())/(values.max()-values.min())
        for channel in ('ArrowRibbon', 'MiniMap'):
            norm, _ = PLOT.make_color_norm(values, channel, contrast_channels=SELECTED)
            np.testing.assert_allclose(norm(values), expected, rtol=0, atol=1e-14)
        default, _ = PLOT.make_color_norm(values, 'StaticSign')
        np.testing.assert_allclose(default(values), expected, rtol=0, atol=1e-14)

    def test_unidentifiable_quantile_anchor_is_rejected(self):
        for values in ([-2.]*10, [-2.]*9+[1.], [-2.]+[1.]*9):
            with self.subTest(values=values), self.assertRaises(ValueError):
                PLOT.make_color_norm(np.asarray(values), 'StaticSign', contrast_channels=SELECTED,
                                     contrast_quantile=.3, contrast_position=.1)

    def test_piecewise_preserves_endpoints_anchor_and_linear_upper_range(self):
        values = np.array([-8., -4., -3., -2., -1.5, -1.2, -1., -.7, -.3, .25])
        before = values.copy()
        lo, hi, anchor = values.min(), values.max(), np.quantile(values, .15)
        for channel in SELECTED:
            with self.subTest(channel=channel):
                norm, metadata = PLOT.make_color_norm(values, channel, contrast_channels=SELECTED,
                    contrast_quantile=.15, contrast_position=.1, contrast_method='piecewise')
                self.assertEqual(metadata['kind'], 'piecewise')
                self.assertIsNone(metadata['gamma'])
                np.testing.assert_allclose(norm([lo, anchor, hi]), [0., .1, 1.], atol=1e-13, rtol=0)
                low = np.linspace(lo, anchor, 21)
                upper = np.linspace(anchor, hi, 21)
                # Equal physical increments retain a constant colour slope inside
                # each segment, especially throughout the newly released upper range.
                np.testing.assert_allclose(np.diff(norm(low))/np.diff(low), .1/(anchor-lo), atol=1e-13, rtol=0)
                np.testing.assert_allclose(np.diff(norm(upper))/np.diff(upper), .9/(hi-anchor), atol=1e-13, rtol=0)
                probes = np.unique(np.r_[low, upper])
                self.assertTrue(np.all(np.diff(norm(probes)) > 0))
                restored = PLOT.norm_from_metadata(metadata)
                np.testing.assert_allclose(restored(probes), norm(probes), atol=1e-14, rtol=0)
                cmap = (PLOT.plt.get_cmap('cividis') if channel == 'StaticSign' else
                        PLOT.LinearSegmentedColormap.from_list('TextBar_piecewise_test', PLOT.PALETTES[channel], N=256))
                np.testing.assert_array_equal(cmap(norm([lo, hi])), cmap([0., 1.]))
        np.testing.assert_array_equal(values, before)

    def test_piecewise_inverse_masks_scalars_and_unclipped_extrapolation(self):
        values = np.array([-8., -4., -3., -2., -1., .25])
        norm, _ = PLOT.make_color_norm(values, 'StaticSign', contrast_channels=SELECTED,
            contrast_quantile=.15, contrast_position=.1, contrast_method='piecewise')
        lo, hi, anchor = values.min(), values.max(), np.quantile(values, .15)
        probes = np.array([lo-2., lo, anchor, 0., hi, hi+2.])
        positions = norm(probes)
        self.assertLess(positions[0], 0.)
        self.assertGreater(positions[-1], 1.)
        np.testing.assert_allclose(norm.inverse(positions), probes, atol=1e-12, rtol=0)
        np.testing.assert_allclose(norm(probes, clip=True), np.clip(positions, 0., 1.), atol=1e-13, rtol=0)
        self.assertTrue(np.isscalar(norm(float(anchor))))
        self.assertTrue(np.isscalar(norm.inverse(.1)))
        self.assertAlmostEqual(float(norm.inverse(.1)), anchor, places=12)
        masked = np.ma.array([lo, anchor, hi], mask=[False, True, False])
        masked_positions = norm(masked)
        restored = norm.inverse(masked_positions)
        np.testing.assert_array_equal(np.ma.getmaskarray(masked_positions), masked.mask)
        np.testing.assert_array_equal(np.ma.getmaskarray(restored), masked.mask)
        np.testing.assert_allclose(restored.compressed(), masked.compressed(), atol=1e-12, rtol=0)

    def test_piecewise_selection_keeps_other_channels_linear_and_power_default(self):
        values = np.array([-8., -4., -3., -2., -1., .25])
        expected_linear = (values-values.min())/(values.max()-values.min())
        for channel in ('ArrowRibbon', 'MiniMap'):
            norm, metadata = PLOT.make_color_norm(values, channel, contrast_channels=SELECTED,
                contrast_quantile=.15, contrast_position=.1, contrast_method='piecewise')
            self.assertEqual(metadata['kind'], 'linear')
            np.testing.assert_allclose(norm(values), expected_linear, atol=1e-14, rtol=0)
        implicit, implicit_meta = PLOT.make_color_norm(values, 'TextBar', contrast_channels=SELECTED)
        explicit, explicit_meta = PLOT.make_color_norm(values, 'TextBar', contrast_channels=SELECTED,
                                                       contrast_method='power')
        self.assertEqual(implicit_meta, explicit_meta)
        self.assertEqual(implicit_meta['kind'], 'power')
        np.testing.assert_array_equal(implicit(values), explicit(values))

    def test_piecewise_rejects_unidentifiable_anchors(self):
        for values in ([-2.]*10, [-2.]*9+[1.], [-2.]+[1.]*9):
            with self.subTest(values=values), self.assertRaises(ValueError):
                PLOT.make_color_norm(np.asarray(values), 'StaticSign', contrast_channels=SELECTED,
                    contrast_quantile=.15, contrast_position=.1, contrast_method='piecewise')


def verify_real_input(path):
    frame = pd.read_csv(path, usecols=['guidance', PLOT.FIELD])
    results = {}
    for channel in PLOT.GUIDANCES:
        values = frame.loc[frame.guidance.eq(channel), PLOT.FIELD].to_numpy(float)
        if not len(values) or not np.isfinite(values).all():
            raise AssertionError('Missing or invalid input values: '+channel)
        norm, metadata = PLOT.make_color_norm(values, channel, contrast_channels=SELECTED,
                                             contrast_quantile=.3, contrast_position=.1)
        lo, hi, q30 = float(values.min()), float(values.max()), float(np.quantile(values, .3))
        np.testing.assert_allclose(norm([lo, hi]), [0., 1.], rtol=0, atol=1e-12)
        if channel in SELECTED:
            gamma = float(np.log(.1)/np.log((q30-lo)/(hi-lo)))
            np.testing.assert_allclose(float(norm(q30)), .1, rtol=0, atol=1e-12)
            np.testing.assert_allclose(norm(np.quantile(values, np.linspace(0, 1, 21))),
                ((np.quantile(values, np.linspace(0, 1, 21))-lo)/(hi-lo))**gamma, rtol=1e-12, atol=1e-12)
        else:
            gamma = 1.
            np.testing.assert_allclose(norm(values), (values-lo)/(hi-lo), rtol=0, atol=1e-12)
        results[channel] = dict(cells=len(values), minimum=lo, maximum=hi, p30=q30,
            p30_color_position=float(norm(q30)), independently_calculated_gamma=gamma)
        gentle, gentle_meta = PLOT.make_color_norm(values, channel, contrast_channels=SELECTED,
            contrast_quantile=.15, contrast_position=.1, contrast_method='piecewise')
        anchor = float(np.quantile(values, .15))
        expected = (np.where(values <= anchor, .1*(values-lo)/(anchor-lo),
                    .1+.9*(values-anchor)/(hi-anchor)) if channel in SELECTED else (values-lo)/(hi-lo))
        np.testing.assert_allclose(gentle(values), expected, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(gentle([lo, hi]), [0., 1.], atol=1e-12, rtol=0)
        levels = np.array([.1, .15, .3, .5, .7, .9])
        quantiles = np.quantile(values, levels)
        if channel in SELECTED:
            self_anchor = float(gentle(anchor))
            np.testing.assert_allclose(self_anchor, .1, atol=1e-12, rtol=0)
            if not float(gentle(np.quantile(values, .5))) > float(norm(np.quantile(values, .5))):
                raise AssertionError('Real-field median is not released relative to old power map: '+channel)
        results[channel]['gentle_mapping'] = dict(kind=gentle_meta['kind'], p15=anchor,
            p15_color_position=float(gentle(anchor)), quantile_levels=levels.tolist(),
            raw_quantiles=quantiles.tolist(), old_color_positions=np.asarray(norm(quantiles)).tolist(),
            new_color_positions=np.asarray(gentle(quantiles)).tolist())
    print(json.dumps(dict(status='passed', input=str(Path(path).resolve()), channels=results),
                     ensure_ascii=False, indent=2))


def main():
    global PLOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--renderer', type=Path, default=ROOT/'0_RL_FRAMEWORK/Plot_Spatial_Rule_3D.py')
    parser.add_argument('--input', type=Path, help='Optional existing exported CSV(.gz) to verify quantile anchors')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('display_renderer_under_test', args.renderer)
    PLOT = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(PLOT)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DisplayRefinement)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1
    if args.input:
        verify_real_input(args.input)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
