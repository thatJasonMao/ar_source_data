"""Regression checks for bounded startup retry and post-training recovery."""
from pathlib import Path
import builtins
import json
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / '0_RL_FRAMEWORK'), str(ROOT / '0_RL_SOURCE'), str(ROOT), str(ROOT / 'TEST')]
import Train_IRL_V6 as train
import Finalize_IRL_V6_Plots as finalize
from Verify_Spatial_Rule_3D import fixture

SCRIPT = ROOT / '0_RL_FRAMEWORK' / 'Plot_Spatial_Rule_3D.py'
FLAG = 'A5_IRL_3D_IMPORT_RETRIED'


def completed_field(best):
    csv, _ = fixture(best)
    (best / 'reward.pt').write_bytes(b'test checkpoint')
    config = json.loads((best / 'config.json').read_text())
    config.update(plot_status='complete', geometry_sha256='test geometry')
    (best / 'config.json').write_text(json.dumps(config), encoding='utf-8')
    files = [g + '_spatial_demand.png' for g in train.GUIDANCE]
    for name in files:
        (csv.parent / name).write_bytes(b'test PNG sentinel')
    (csv.parent / 'manifest.json').write_text(json.dumps(dict(status='complete',
        method=config['method'], files=files, geometry_sha256=config['geometry_sha256'],
        raw_csv_sha256=train.digest(csv), reward_sha256=train.digest(best / 'reward.pt'))), encoding='utf-8')
    return csv


class StartupRecovery(unittest.TestCase):
    def execute_with_import_failure(self, message, already_retried=False):
        original_import = builtins.__import__
        def fail_pyplot(name, *args, **kwargs):
            if name == 'matplotlib.pyplot':
                raise RuntimeError(message)
            return original_import(name, *args, **kwargs)
        with patch.dict(os.environ):
            os.environ.pop(FLAG, None)
            if already_retried:
                os.environ[FLAG] = '1'
            with patch.object(sys, 'argv', [str(SCRIPT), '--help']), \
                    patch.object(builtins, '__import__', side_effect=fail_pyplot), \
                    patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as execute:
                try:
                    runpy.run_path(str(SCRIPT), run_name='__main__')
                except BaseException as exc:
                    return exc, execute.call_args_list
        self.fail('Expected startup failure or child exit')

    def test_known_error_restarts_exact_arguments_once(self):
        exc, calls = self.execute_with_import_failure('internal error in regular expression engine')
        self.assertIsInstance(exc, SystemExit)
        self.assertEqual(exc.code, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args[0], [sys.executable, '-B', str(SCRIPT), '--help'])
        self.assertEqual(calls[0].kwargs['env'][FLAG], '1')

    def test_second_failure_does_not_restart_again(self):
        exc, calls = self.execute_with_import_failure('internal error in regular expression engine', True)
        self.assertIsInstance(exc, RuntimeError)
        self.assertFalse(calls)

    def test_other_import_errors_are_not_retried(self):
        exc, calls = self.execute_with_import_failure('unrelated import error')
        self.assertIsInstance(exc, RuntimeError)
        self.assertFalse(calls)


class PlotRecovery(unittest.TestCase):
    def test_resume_reuses_verified_2d_without_touching_raw_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            best = Path(tmp) / 'Best_Model'
            csv = completed_field(best)
            before = train.digest(csv)
            with patch.object(train.subprocess, 'run') as execute:
                train.plot_best_model(best)
            execute.assert_called_once_with([sys.executable, '-B', str(SCRIPT), '--run', str(best)], check=True)
            self.assertEqual(train.digest(csv), before)
            self.assertEqual(json.loads((best / 'config.json').read_text())['plot_3d_status'], 'complete')

    def test_corrupt_field_is_rejected_before_rendering(self):
        with tempfile.TemporaryDirectory() as tmp:
            best = Path(tmp) / 'Best_Model'
            csv = completed_field(best)
            csv.write_bytes(b'changed input')
            with patch.object(train.subprocess, 'run') as execute:
                with self.assertRaisesRegex(ValueError, 'integrity checks'):
                    train.plot_best_model(best)
            execute.assert_not_called()

    def test_failed_recovery_never_marks_run_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / '20261003_020827_592825_V6_1'
            best = run / 'Best_Model'
            completed_field(best)
            (run / 'config.json').write_text(json.dumps(dict(status='running')), encoding='utf-8')
            trial = run / 'Trials' / 'selected'
            trial.mkdir(parents=True)
            (trial / 'reward.pt').write_bytes((best / 'reward.pt').read_bytes())
            (run / 'selection.json').write_text(json.dumps(dict(candidate=dict(output=str(trial)))), encoding='utf-8')
            config = json.loads((best / 'config.json').read_text())
            config['active_guidance'] = train.GUIDANCE
            (best / 'config.json').write_text(json.dumps(config), encoding='utf-8')
            for name in ['reward_basis.json', 'encoder.json', 'mdp.npz', 'state_reward_features.npy',
                         'reward_rules.csv', 'training_history.csv', 'evaluation_identity.json', 'evaluation_predictions.npz']:
                (best / name).write_bytes(b'numerical export sentinel')
            (best / 'evaluation.json').write_text(json.dumps(dict(trained=True, train={}, val={}, test={})), encoding='utf-8')
            with patch.object(finalize, 'plot_best_model', side_effect=RuntimeError('render failure')):
                with self.assertRaisesRegex(RuntimeError, 'render failure'):
                    finalize.finalize_run(run)
            self.assertFalse((run / 'completion.json').exists())
            self.assertEqual(json.loads((run / 'config.json').read_text())['status'], 'running')


if __name__ == '__main__':
    unittest.main(verbosity=2)
