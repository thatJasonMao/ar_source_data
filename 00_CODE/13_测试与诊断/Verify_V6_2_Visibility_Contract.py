"""Read-only V6.2 visibility contract tests; never fit or train a model.

The 10/14 query test temporarily relabels an in-memory copy of an OLD model
to test radius dispatch only. Its utility values are NOT newly fitted results.
"""
from pathlib import Path
import argparse
import ast
import copy
import json
import sys
import unittest
from unittest.mock import patch


def ast_audit(before, after):
    def module_contract(path):
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        body = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, ast.ImportFrom) and node.module == 'spatial_isovist_core':
                if [(n.name, n.asname) for n in node.names] == [('SMOKE_VISIBILITY_M', None)]:
                    continue
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                if isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('VISIBILITY', 'LEGACY_VISIBILITY'):
                    continue
            body.append(node)
        return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)

    def definitions(path):
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        result = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result[node.name] = ast.dump(node, include_attributes=False)
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        result[node.name + '.' + child.name] = ast.dump(child, include_attributes=False)
                body = [n for n in node.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                result[node.name + '.__class_contract__'] = ast.dump(ast.ClassDef(
                    name=node.name, bases=node.bases, keywords=node.keywords, body=body,
                    decorator_list=node.decorator_list), include_attributes=False)
        return result

    old, new = definitions(before), definitions(after)
    if module_contract(before) != module_contract(after):
        raise AssertionError('Non-visibility module configuration changed')
    changed = sorted(k for k in old.keys() & new.keys() if old[k] != new[k])
    added, removed = sorted(new.keys() - old.keys()), sorted(old.keys() - new.keys())
    allowed = {'audit_scope', 'validate_trial', 'TrialReader.__init__', 'TrialReader.fixed_geometry',
        'cache_for', 'load_cache', 'export_model', 'SpatialRule.__init__', 'SpatialRule.query_features',
        'SpatialRule.query_position', 'build_grid', 'export_fields', 'main'}
    if set(changed) - allowed or set(added) != {'effective_visibility'} or removed:
        raise AssertionError({'unexpected_changes': sorted(set(changed) - allowed),
                              'added': added, 'removed': removed})
    protected = ['Reward.__init__', 'Reward.forward', 'build_mdp', 'train_candidate', 'search_channel',
                 'candidate_configs', 'frame_logits', 'frame_report', 'rank', 'select', 'pack_frames',
                 'Encoder.__init__', 'Encoder.names', 'Encoder.transform', 'load_split', 'context_keys']
    for name in protected:
        if old[name] != new[name]:
            raise AssertionError('Training implementation changed: ' + name)
    return dict(passed=True, changed_visibility_functions=changed, added_visibility_functions=added,
                nonvisibility_module_configuration_unchanged=True,
                unchanged_definition_count=sum(old[k] == new[k] for k in old.keys() & new.keys()),
                protected_training_definitions_unchanged=protected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--before', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    for p in (root, root / '0_RL_SOURCE', root / '0_RL_FRAMEWORK'):
        sys.path.insert(0, str(p))
    import Train_IRL_V6_2 as model
    from spatial_isovist_core import SMOKE_VISIBILITY_M, PlanarIsovist

    original = args.before or root / 'Documentation/Visibility_Calibration_20261108/before/0_RL_FRAMEWORK/Train_IRL_V6_2.py'
    audit = ast_audit(original, root / '0_RL_FRAMEWORK/Train_IRL_V6_2.py')
    legacy_dataset = root / '0_RL_SOURCE/Results_IRL_Dataset_V6/20261003_010150_548829'
    legacy_model = root / '0_RL_FRAMEWORK/Results_IRL/20261003_230129_557741_V6_2/Best_Model'
    report = {}

    class VisibilityTests(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.guards = []
            blocked = AssertionError('This audit must not fit, train, build an MDP, or launch workers')
            for owner, name in [(model, n) for n in ('train_candidate', 'search_channel', 'build_mdp',
                    'cache_for', 'export_model', 'load_split', 'launch_batch')] + [
                    (model.StateEncoder, 'fit'), (model.SpatialRewardBasis, 'fit'),
                    (model.MiniBatchKMeans, 'fit'), (model.torch.optim.Adam, 'step'),
                    (model.subprocess, 'Popen')]:
                guard = patch.object(owner, name, side_effect=blocked)
                cls.guards.append(guard)
                guard.start()
            cls.addClassCleanup(lambda: [guard.stop() for guard in reversed(cls.guards)])
            cls.bank = model.SpatialRule(legacy_model)

        def simulated_current_bank(self):
            bank = copy.copy(self.bank)
            bank.config = dict(bank.config, smoke_visibility_m=dict(model.VISIBILITY))
            bank.visibility = dict(model.VISIBILITY)
            bank.models = {}
            for name, old_child in self.bank.models.items():
                child = copy.copy(old_child)
                child.config = dict(child.config, smoke_visibility_m=dict(model.VISIBILITY))
                child.visibility = dict(model.VISIBILITY)
                bank.models[name] = child
            return bank

        def test_current_mapping(self):
            self.assertEqual(model.VISIBILITY, {'HighSmoke': 10.0, 'LowSmoke': 14.0})
            self.assertEqual(model.VISIBILITY, SMOKE_VISIBILITY_M)

        def test_old_dataset_rejected_by_plan_and_reader(self):
            with self.assertRaisesRegex(ValueError, 'rebuilt dataset'):
                model.main(['--dataset', str(legacy_dataset)])
            manifest = model.read_json(legacy_dataset / 'manifest.json')
            with self.assertRaisesRegex(ValueError, 'source visibility'):
                model.TrialReader(legacy_dataset, manifest, model.Profile(), root / 'unused', {})

        def test_v6_3_fixed_radius_unchanged(self):
            profile = model.Profile('V6_3', True, 12.5)
            self.assertEqual(model.effective_visibility(profile), {'PooledSmoke': 12.5})
            _, scope = model.audit_scope(legacy_dataset, profile)
            self.assertEqual(scope['selected_trials'], 178)

        def test_legacy_model_point_queries_keep_historical_radii(self):
            values = {}
            for smoke, radius in {'HighSmoke': 7.5, 'LowSmoke': 12.5}.items():
                result = self.bank.query_position(0, -10, 0, 'StaticSign', smoke=smoke)
                self.assertEqual(result['status'], 'ok')
                self.assertEqual(result['visibility_m'], radius)
                values[smoke] = radius
            report['verified_legacy_model_radii'] = values

        def test_new_metadata_rejects_wrong_and_missing_feature_radius(self):
            bank = self.simulated_current_bank()
            result = bank.query_position(0, -10, 0, 'StaticSign', smoke='HighSmoke')
            frame = model.pd.DataFrame([result])
            with self.assertRaisesRegex(ValueError, 'saved model visibility'):
                bank.query_features(frame.drop(columns='visibility_m'), 'StaticSign')
            frame['visibility_m'] = 7.5
            with self.assertRaisesRegex(ValueError, 'saved model visibility'):
                bank.query_features(frame, 'StaticSign')

        def test_simulated_new_metadata_dispatches_10_14(self):
            bank = self.simulated_current_bank()
            engine = PlanarIsovist(bank.config['geometry']['path'])
            values = {}
            for smoke, radius in model.VISIBILITY.items():
                result = bank.query_position(0, -10, 0, 'StaticSign', smoke=smoke)
                expected = engine.query(0, -10, 0, radius)
                self.assertEqual(result['status'], 'ok')
                self.assertEqual(result['visibility_m'], radius)
                for name in model.SPATIAL[:4]:
                    self.assertAlmostEqual(result[name], getattr(expected, name), places=10)
                values[smoke] = radius
            report['simulated_new_metadata_radii'] = values
            report['simulation_scope'] = 'radius dispatch only; old weights in memory; no newly fitted utility results'

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(VisibilityTests))
    print(json.dumps(dict(passed=result.wasSuccessful(), tests_run=result.testsRun,
                         optimizer_executed=False, model_fitted=False, ast_audit=audit, **report), indent=2))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
