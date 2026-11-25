"""Rendering and V6 export integration checks; no model fitting or training."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / '0_RL_FRAMEWORK'), str(ROOT / '0_RL_SOURCE'), str(ROOT)]
import Plot_Spatial_Rule_3D as plot
import Train_IRL_V6 as train


def fixture(run, channels=plot.GUIDANCES):
    run.mkdir(parents=True, exist_ok=True)
    (run / 'config.json').write_text(json.dumps(dict(status='complete', version=6,
        method='four_channel_full_frame_spatial_irl_v6', plot_status='pending')), encoding='utf-8')
    field_dir = run / 'Results_Global_Spatial_Field_V6'
    field_dir.mkdir(exist_ok=True)
    # Four observed cells plus a deliberate hole; values include negatives/zero.
    frame = pd.DataFrame([dict(guidance=g, smoke=s, floor=f, x=(ix+.5)*.4,
        z=(iz+.5)*.4, relative_information_utility=value*(gi+1))
        for gi, g in enumerate(channels) for s in plot.SMOKES for f in plot.FLOORS
        for ix, iz, value in [(0, 0, -2.), (1, 0, 0.), (0, 1, 1.), (2, 1, 3.)]])
    csv = field_dir / plot.CSV_NAME
    frame.to_csv(csv, index=False, compression='gzip')
    return csv, frame


class SpatialRendering(unittest.TestCase):
    def test_v6_requires_static_even_when_other_three_are_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'Best_Model'
            csv, _ = fixture(run, plot.LEGACY_GUIDANCES)
            source, channels = plot.resolve_input(run=run)
            self.assertEqual(source, csv)
            with self.assertRaisesRegex(ValueError, 'all 16'):
                plot.load_data(source, channels)
            self.assertEqual(set(plot.load_data(csv).guidance), set(plot.LEGACY_GUIDANCES))

    def test_input_validation_and_exact_cell_heights(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv, frame = fixture(Path(tmp) / 'Best_Model')
            loaded = plot.load_data(csv)
            part = loaded.loc[loaded.guidance.eq('StaticSign') &
                              loaded.smoke.eq('HighSmoke') & loaded.floor.eq('Hall')]
            faces, values = plot.cell_faces(part)
            self.assertEqual(len(faces), 4)  # No cell inserted into the hole.
            np.testing.assert_array_equal(values, [-2., 0., 1., 3.])
            np.testing.assert_array_equal(faces[:, :, 2], np.repeat(values[:, None], 4, axis=1))
            np.testing.assert_allclose(faces[:, :, 0].mean(axis=1), -part.x.to_numpy())
            np.testing.assert_allclose(faces[:, :, 1].mean(axis=1), -part.z.to_numpy())
            for invalid, message in [(pd.concat([frame, frame.iloc[:1]]), 'Duplicate'),
                                     (frame.assign(x=frame.x+.01), 'cell-centre'),
                                     (frame.assign(relative_information_utility=np.nan), 'invalid')]:
                invalid.to_csv(csv, index=False, compression='gzip')
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    plot.load_data(csv)

    def test_real_four_channel_render_manifest_and_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'Best_Model'
            csv, frame = fixture(run)
            original_hash = plot.sha256(csv)
            manifest = plot.plot_field(run=run, dpi=50)
            out = csv.parent / 'Results_3D'
            self.assertEqual(len(list(out.glob('*.png'))), 8)
            self.assertEqual(manifest['guidances'], list(plot.GUIDANCES))
            self.assertEqual(manifest['rows_rendered'], len(frame))
            self.assertEqual(manifest['input_sha256'], original_hash)
            self.assertEqual(plot.sha256(csv), original_hash)
            for g in plot.GUIDANCES:
                m3, m2 = manifest['figures'][g], manifest['figures_2d'][g]
                self.assertEqual(m3['panels'], m2['panels'])
                self.assertEqual(m3['cmap_minimum'], m3['data_minimum'])
                self.assertEqual(m3['cmap_maximum'], m3['data_maximum'])
                self.assertTrue(all(p['cells'] == 4 for p in m3['panels'].values()))
                self.assertGreater((out / m3['file']).stat().st_size, 1000)
            with patch.object(plot, 'render_guidance', side_effect=AssertionError('Unneeded rerender')):
                self.assertEqual(plot.plot_field(run=run, dpi=50), manifest)
                with self.assertRaises(FileExistsError):
                    plot.plot_field(run=run, dpi=60)
            # A failed replacement cannot leave an apparently complete manifest.
            with patch.object(plot, 'render_guidance', side_effect=RuntimeError('render failure')):
                with self.assertRaises(RuntimeError):
                    plot.plot_field(run=run, dpi=50, force=True)
            self.assertNotEqual(json.loads((out / 'manifest_3d.json').read_text())['status'], 'complete')

    def test_completed_legacy_run_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'Best_Model'
            csv, _ = fixture(run, plot.LEGACY_GUIDANCES)
            legacy = run / 'Results_Global_Spatial_Field_V4'
            csv.parent.rename(legacy)
            (run / 'config.json').write_text(json.dumps(dict(status='complete',
                method='budgeted_full_frame_spatial_IRL_v4')), encoding='utf-8')
            source, channels = plot.resolve_input(run=run)
            self.assertEqual(source, legacy / plot.CSV_NAME)
            self.assertEqual(channels, plot.LEGACY_GUIDANCES)
            self.assertEqual(len(plot.load_data(source, channels)), 48)


class BestModelIntegration(unittest.TestCase):
    def test_sequential_2d_then_3d_and_complete_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'Best_Model'
            fixture(run)
            calls = []
            def execute(command, check):
                self.assertTrue(check)
                self.assertEqual(command[-2:], ['--run', str(run.resolve())])
                name = Path(command[2]).name
                calls.append(name)
                if name == 'Plot_IRL_V6.py':
                    config = json.loads((run / 'config.json').read_text())
                    config['plot_status'] = 'complete'
                    (run / 'config.json').write_text(json.dumps(config), encoding='utf-8')
                else:
                    self.assertEqual(json.loads((run / 'config.json').read_text())['plot_status'], 'complete')
            with patch.object(train.subprocess, 'run', side_effect=execute):
                train.plot_best_model(run)
            self.assertEqual(calls, ['Plot_IRL_V6.py', 'Plot_Spatial_Rule_3D.py'])
            config = json.loads((run / 'config.json').read_text())
            self.assertEqual(config['plot_status'], 'complete')
            self.assertEqual(config['plot_3d_status'], 'complete')

    def test_skip_plots_does_not_launch_any_renderer(self):
        with patch.object(train.subprocess, 'run') as execute:
            train.plot_best_model(Path('unused'), skip_plots=True)
            execute.assert_not_called()

    def test_render_failure_is_reported_without_removing_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'Best_Model'
            fixture(run)
            sentinel = run / 'reward.pt'
            sentinel.write_bytes(b'test sentinel, not a trained model')
            failure = subprocess.CalledProcessError(1, '3D renderer')
            with patch.object(train.subprocess, 'run', side_effect=[None, failure]):
                with self.assertRaises(subprocess.CalledProcessError):
                    train.plot_best_model(run)
            config = json.loads((run / 'config.json').read_text())
            self.assertEqual(config['plot_3d_status'], 'failed')
            self.assertIn('3D renderer', config['plot_3d_error'])
            self.assertEqual(sentinel.read_bytes(), b'test sentinel, not a trained model')


if __name__ == '__main__':
    unittest.main(verbosity=2)
