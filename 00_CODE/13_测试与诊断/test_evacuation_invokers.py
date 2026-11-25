"""Invoker contract checks using a synthetic common.load_trials() fixture."""
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def fixture(metric):
    records = []
    for group, mode in [('A', 'ArrowRibbon'), ('B', 'MiniMap'), ('C', 'TextBar')]:
        for number in range(1, 16):
            for guidance in ['StaticSign', mode, 'Full']:
                for smoke in ['HighSmoke', 'LowSmoke']:
                    scenario = f'{guidance}-{smoke}'
                    records.append(dict(Scenario=scenario, ParticipantID=f'P_{group}{number}',
                                        FileName=f'{scenario}_{number}.csv', Group=group,
                                        Guidance=guidance, Smoke=smoke))
    trials = pd.DataFrame(records)
    raw = trials[['Scenario', 'ParticipantID', 'FileName']].copy()
    raw['Subject'] = raw.ParticipantID
    raw[metric] = np.random.default_rng(41).uniform(5, 20, len(raw))
    raw['PlotIncluded'] = False
    return raw, SimpleNamespace(load_trials=lambda: trials.copy())


class ContractChecks:
    modules = []

    def test_exact_cohort_grouping_and_pairing(self):
        for name in self.modules:
            with self.subTest(module=name):
                module = import_module('INDICATORS_SIGNIFICANCE.Invoker_' + name)
                raw, common = fixture(module.METRIC)
                data, audit, evidence = module.prepare_data(raw.sample(frac=1, random_state=7), common)
                self.assertEqual(len(data), 270)
                self.assertEqual(data.SubjectID.nunique(), 45)
                self.assertTrue(data.SessionCount.eq(1).all())
                self.assertTrue(audit.Status.eq('included').all())
                self.assertEqual(evidence, [])
                jobs = module.build_jobs(data, .05)
                self.assertEqual(len(jobs), 7)
                for job in jobs[:5]:
                    self.assertTrue(job['unmatched'].empty)
                    self.assertEqual(job['subject_ids']['HighSmoke'], job['subject_ids']['LowSmoke'])
                self.assertEqual([len(v) for v in jobs[-1]['groups'].values()], [15, 15, 15])

    def test_bad_inputs_fail_instead_of_reducing_cohort(self):
        for name in self.modules:
            module = import_module('INDICATORS_SIGNIFICANCE.Invoker_' + name)
            raw, common = fixture(module.METRIC)
            variants = [raw.iloc[:-1], pd.concat([raw, raw.iloc[:1]], ignore_index=True)]
            changed = raw.copy()
            changed.loc[0, ['Subject', 'ParticipantID']] = 'TEST'
            variants.append(changed)
            changed = raw.copy()
            changed.loc[0, 'FileName'] = 'other_session.csv'
            variants.append(changed)
            for value in [np.nan, np.inf, -1]:
                changed = raw.copy()
                changed.loc[0, module.METRIC] = value
                variants.append(changed)
            changed = raw.copy()
            changed['SubjectGroup'] = 'WRONG'
            variants.append(changed)
            for i, frame in enumerate(variants):
                with self.subTest(module=name, case=i), self.assertRaises(ValueError):
                    module.prepare_data(frame, common)


class MetricInvokerTests(ContractChecks, unittest.TestCase):
    modules = ['EvacuationRotation', 'EvacuationSpeed', 'EvacuationTime']


if __name__ == '__main__':
    unittest.main()
