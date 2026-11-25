"""V6 contract/regression tests. No optimizer steps or real-data fitting."""
from pathlib import Path
import sys
import ast
import json
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / '0_RL_SOURCE'), str(ROOT / '0_RL_FRAMEWORK')]
from Guidance_Channel_Contract import CHANNELS, FIELDS, static_arrays, channel_actions
from Query_GuidanceGaze import event_flags
from Build_IRL_Dataset_V6 import attach_channels
from irl_data_v6 import trial_masses, assign_states, StateEncoder, NUMERIC
from irl_frame_optimization_v6 import frame_logit, frame_report
from irl_core_v3 import LinearReward, soft_policy
from spatial_reward_basis import SpatialRewardBasis
from Spatial_IRL_Rule_V6 import SpatialIRLRule, SCOPE, reward_mask


class Contracts(unittest.TestCase):
    def test_static_names_and_invalid_coordinates(self):
        names = ['ABCD_Signal', 'ABCD_Signal (1)', 'ABCD_Signal (13)', 'ABCD_Signal (14)',
                 'ABCD_Signal (0)', 'prefixABCD_Signal (1)', 'ABCD_Signal (2)']
        eye = pd.DataFrame(dict(Hit1_Name=names, Hit1_X=[1] * 6 + [0], Hit1_Y=[2] * 6 + [0], Hit1_Z=[3] * 6 + [0]))
        valid, hits = static_arrays(eye)
        np.testing.assert_array_equal(hits, [1, 1, 1, 0, 0, 0, 0])
        self.assertFalse(valid[-1])

    def test_absence_is_not_zero_and_overlap_is_multilabel(self):
        f = pd.DataFrame({v: [1, 1] for v, _ in FIELDS.values()})
        for _, g in FIELDS.values():
            f[g] = [1, 0]
        for c in CHANNELS:
            np.testing.assert_array_equal(channel_actions(f, 'Full', c), [1, 0])
        np.testing.assert_array_equal(channel_actions(f, 'StaticSign', 'MiniMap'), [-1, -1])
        f.loc[1, 'ValidStaticSignAOI'] = 0
        np.testing.assert_array_equal(channel_actions(f, 'Full', 'StaticSign'), [1, -1])

    def test_raw_hits_and_events_differ(self):
        ns = np.arange(8, dtype=np.int64) * 22_000_000
        one = np.array([0, 1, 0, 0, 0, 0, 0, 0], bool)
        self.assertEqual(one.sum(), 1)
        self.assertFalse(event_flags(ns, one)[0].any())
        run = np.array([0, 1, 1, 1, 1, 1, 0, 0], bool)
        np.testing.assert_array_equal(event_flags(ns, run)[0], run)

    def test_channel_history_gaps_and_exit_masks(self):
        n = 7
        ns = np.array([0, 22, 44, 66, 300, 322, 344], dtype=np.int64) * 1_000_000
        df = pd.DataFrame(dict(timestamp_ns=ns, dt_next_s=np.r_[np.diff(ns) / 1e9, np.nan],
            x=np.zeros(n), y=np.zeros(n), z=np.zeros(n), speed=np.ones(n), speed_prev=np.ones(n),
            stage=['to_gate'] * 6 + ['arrived_exit'], spatial_status=['ok'] * n,
            decision_status=['ok'] * n, slow_state=np.zeros(n), split=['train'] * n))
        actions = {'StaticSign': np.array([1, 0, -1, 1, 1, 0, 1]), 'MiniMap': np.array([0, 1, 1, 1, 0, 1, 0])}
        class Query:
            def query_frames(self, row, channel, stamps):
                return actions.get(channel, np.full(n, -1))
        row = SimpleNamespace(Guidance='MiniMap', trial_id='t0', ParticipantID='p0', Scenario='MiniMap-HighSmoke', Smoke='HighSmoke')
        out, runs, audit = attach_channels(df, row, Query(), SimpleNamespace(max_gap_s=.1, run_cap_s=2.))
        self.assertEqual(out.previous_action_StaticSign.iloc[3], -1)
        self.assertEqual(out.previous_action_MiniMap.iloc[3], 1)
        self.assertEqual(out.previous_action_MiniMap.iloc[4], -1)
        self.assertFalse(out.transition_valid_StaticSign.iloc[1])
        self.assertFalse(out.valid_MiniMap.iloc[-1])
        self.assertTrue(out.action_ArrowRibbon.eq(-1).all())
        self.assertEqual({r['channel'] for r in runs}, {'StaticSign', 'MiniMap'})

    def test_weighting_does_not_count_full_four_times(self):
        records = []
        for c in CHANNELS:
            for condition in [c, 'Full']:
                for i in range(1 if condition == c else 3):
                    records.append(dict(trial_id=f'{c}/{condition}/{i}', guidance=c,
                        condition=condition, participant_id=f'p{i}', valid_rows=10))
        mass = trial_masses(records)
        self.assertAlmostEqual(sum(mass.values()), 1)
        for c in CHANNELS:
            self.assertAlmostEqual(sum(mass[r['trial_id']] for r in records if r['guidance'] == c), .25)
            self.assertAlmostEqual(sum(mass[r['trial_id']] for r in records if r['guidance'] == c and r['condition'] == 'Full'), .125)

    def test_state_fallback_stays_in_condition(self):
        df = pd.DataFrame(dict(guidance=['StaticSign'] * 2, condition=['Full', 'MiniMap'],
            smoke=['HighSmoke'] * 2, previous_action=[1, 1]))
        states, _ = assign_states(df, np.array([[0.], [0.]]), {'StaticSign/Full/HighSmoke/0': (0, 1)}, np.array([[0.]]))
        np.testing.assert_array_equal(states, [0, -1])

    def test_forward_matches_v3_at_state_means(self):
        torch.manual_seed(1)
        model = LinearReward(3, n_guidance=4)
        with torch.no_grad():
            model.shared.copy_(torch.tensor([.1, .2, -.1]))
            model.deviation.copy_(torch.arange(12).reshape(4, 3) * .01)
        x = torch.tensor([[1., 1., 0.], [1., 0., 1.], [0., 0., 0.]])
        guide = torch.tensor([0, 1, 0]); state = torch.tensor([0, 1])
        trans = torch.zeros((3, 2, 3)); trans[:, :, 2] = 1
        dt = torch.ones((3, 2)) * .022
        mdp = dict(guide_tensor=guide, terminal_index=2)
        z, _ = frame_logit(model, x[:2], guide[:2], state, x, mdp, trans, dt, 3, .98)
        _, q = soft_policy(model(x, guide), trans, 3, .98, 2, mean_dt=dt)
        torch.testing.assert_close(z, (q[:, 1] - q[:, 0])[:2])
        z.sum().backward()
        self.assertTrue(torch.isfinite(model.shared.grad).all())
        self.assertTrue(torch.isfinite(model.deviation.grad).all())

    def test_channel_export_load_and_order(self):
        df = pd.DataFrame({c: [1., 2.] for c in NUMERIC})
        df['previous_action'] = [-1, 0]; df['floor'] = ['Hall', 'Platform']
        df['stage'] = ['to_gate', 'to_stair']; df['smoke'] = ['HighSmoke', 'LowSmoke']
        encoder = StateEncoder().fit(df)
        mask = reward_mask(encoder.names).astype(bool)
        x = encoder.transform(df)[:, mask]
        names = [n for n, keep in zip(encoder.names, mask) if keep]
        basis = SpatialRewardBasis.fit(x, names, 'linear')
        model = LinearReward(len(basis.names), n_guidance=4)
        with torch.no_grad():
            model.deviation[:, 0] = torch.tensor([-1., 0., 1., 2.])
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            p = Path(tmp)
            (p / 'config.json').write_text(json.dumps(dict(status='complete', reward_scope=SCOPE, version=6,
                method='four_channel_full_frame_spatial_irl_v6', active_guidance=CHANNELS)))
            (p / 'encoder.json').write_text(json.dumps(dict(encoder=encoder.as_dict())))
            (p / 'reward_basis.json').write_text(json.dumps(basis.spec))
            torch.save(model.state_dict(), p / 'reward.pt')
            rule = SpatialIRLRule(p)
            values = [rule.query_features(df, c)[0] for c in CHANNELS]
            np.testing.assert_allclose(np.diff(values), np.ones(3), atol=1e-6)
            with self.assertRaises(ValueError):
                rule.query_features(df, 'Full')

    def test_cli_options_match_v3_optimize(self):
        def options(path):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            return {n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument'
                and n.args and isinstance(n.args[0], ast.Constant)}
        self.assertEqual(options(ROOT / '0_RL_FRAMEWORK/Train_IRL_V3_Optimize.py'), options(ROOT / '0_RL_FRAMEWORK/Train_IRL_V6.py'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
