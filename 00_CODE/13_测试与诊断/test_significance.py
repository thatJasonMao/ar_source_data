"""Synthetic numerical/contract tests; never reads experiment data."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from INDICATORS_SIGNIFICANCE import analyze_significance, write_report
from INDICATORS_SIGNIFICANCE.Analyze_Significance import _dunn


class SignificanceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(812)
        base = rng.normal(size=30)
        self.groups = {name: base + rng.normal(size=30) + i * 1.5
                       for i, name in enumerate('ABC')}

    def run_test(self, **kw):
        args = dict(groups=self.groups, distribution='parametric', design='independent', equal_var=True)
        args.update(kw)
        return analyze_significance(**args)

    def test_all_dispatch_branches(self):
        for k in (2, 3):
            for dist in ('parametric', 'nonparametric'):
                for design in ('independent', 'repeated'):
                    for eq in (True, False):
                        with self.subTest(k=k, dist=dist, design=design, eq=eq):
                            r = self.run_test(groups=dict(list(self.groups.items())[:k]), distribution=dist,
                                              design=design, equal_var=eq, assume_aligned=True, posthoc='always')
                            self.assertTrue(0 <= r['omnibus']['p_value'] <= 1)
                            self.assertEqual(len(r['posthoc']), 3 if k == 3 else 0)
                            for row in r['posthoc']:
                                self.assertTrue(0 <= row['p_adjusted'] <= 1)
                                if 'p_value' in row:
                                    self.assertGreaterEqual(row['p_adjusted'] + 1e-14, row['p_value'])

    def test_independent_numerics(self):
        a, b, c = self.groups.values()
        self.assertAlmostEqual(self.run_test()['omnibus']['p_value'], stats.f_oneway(a, b, c).pvalue)
        for eq in (True, False):
            r = self.run_test(groups={'A': a, 'B': b}, equal_var=eq)
            self.assertAlmostEqual(r['omnibus']['p_value'], stats.ttest_ind(a, b, equal_var=eq).pvalue)
        r = self.run_test(distribution='nonparametric')
        self.assertAlmostEqual(r['omnibus']['p_value'], stats.kruskal(a, b, c).pvalue)

    def test_rm_against_statsmodels(self):
        x = np.column_stack(list(self.groups.values()))
        frame = pd.DataFrame({'subject': np.repeat(np.arange(len(x)), 3),
                              'condition': np.tile(['A', 'B', 'C'], len(x)), 'value': x.ravel()})
        frame['condition'] = frame['condition'].astype(object)
        reference = AnovaRM(frame, 'value', 'subject', within=['condition']).fit().anova_table.iloc[0]
        r = self.run_test(design='repeated', assume_aligned=True, sphericity=True)['omnibus']
        self.assertAlmostEqual(r['statistic'], reference['F Value'], places=10)
        self.assertAlmostEqual(r['p_value'], reference['Pr > F'], places=12)
        gg = self.run_test(design='repeated', assume_aligned=True)['omnibus']
        # Independent contrast/eigenvalue formulation of GG epsilon.
        h = np.array([[1, -1, 0], [1, 1, -2]], dtype=float)
        h /= np.linalg.norm(h, axis=1)[:, None]
        eigen = np.linalg.eigvalsh(h @ np.cov(x, rowvar=False) @ h.T)
        expected = eigen.sum() ** 2 / (2 * np.sum(eigen ** 2))
        self.assertAlmostEqual(gg['epsilon_gg'], expected)
        self.assertAlmostEqual(gg['p_value'], stats.f.sf(r['statistic'], 2 * expected, 58 * expected))
        self.assertGreaterEqual(gg['p_value'], r['p_value'])

    def test_dunn_hand_calculation(self):
        rows = _dunn([np.array([1, 2]), np.array([3, 4]), np.array([5, 6])])
        z = (1.5 - 3.5) / np.sqrt(6 * 7 / 12 * (1 / 2 + 1 / 2))
        self.assertAlmostEqual(rows[0][2]['statistic'], z)
        self.assertAlmostEqual(rows[0][2]['p_value'], 2 * stats.norm.sf(abs(z)))
        tied = _dunn([np.array([1, 1]), np.array([2, 3]), np.array([3, 4])])
        variance = 6 * 7 / 12 - 12 / (12 * 5)
        self.assertAlmostEqual(tied[0][2]['statistic'], (1.5 - 3.75) / np.sqrt(variance))

    def test_pair_alignment_and_missing(self):
        g = {'A': [1, None, 3, 6], 'B': [2, 4, 5, 8]}
        ids = {'A': ['a', 'b', 'c', 'd'], 'B': ['a', 'b', 'c', 'd']}
        r = self.run_test(groups=g, design='repeated', subject_ids=ids, missing='omit')
        self.assertEqual(r['retained_subject_ids'], ['a', 'c', 'd'])
        self.assertEqual([d['n_removed'] for d in r['descriptive']], [1, 1])
        self.assertAlmostEqual(r['omnibus']['p_value'], stats.ttest_rel([1, 3, 6], [2, 5, 8]).pvalue)
        ids['B'] = ['b', 'a', 'c', 'd']
        with self.assertRaises(ValueError):
            self.run_test(groups=g, design='repeated', subject_ids=ids, missing='omit')

    def test_invalid_inputs(self):
        cases = [dict(group_count=4), dict(distribution='unknown'), dict(design='mixed'),
                 dict(equal_var=None), dict(alpha=1), dict(alpha=True), dict(missing='silent'),
                 dict(groups={'A': [1], 'B': [2]}), dict(groups={'A': [1, np.inf], 'B': [2, 3]}),
                 dict(groups={'A': [1, None], 'B': [2, 3]}), dict(design='repeated'),
                 dict(groups={'A': [1, 2], 'B': [1, 2, 3]}, design='repeated', assume_aligned=True)]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.run_test(**case)

    def test_degenerate_and_scaling(self):
        r = self.run_test(groups={'A': [1, 2, 3], 'B': [1, 2, 3]}, distribution='nonparametric',
                          design='repeated', assume_aligned=True)
        self.assertEqual(r['omnibus']['p_value'], 1)
        with self.assertRaises(ValueError):
            self.run_test(groups={'A': [1, 1], 'B': [1, 1]})
        original = self.run_test(design='repeated', assume_aligned=True)['omnibus']
        scaled = self.run_test(groups={k: v * 1e-10 for k, v in self.groups.items()},
                               design='repeated', assume_aligned=True)['omnibus']
        self.assertAlmostEqual(original['statistic'], scaled['statistic'])

    def test_posthoc_policy_and_reports(self):
        r = self.run_test(posthoc='none')
        self.assertEqual(r['posthoc_status'], 'disabled')
        nonsig = self.run_test(groups={k: self.groups['A'] for k in 'ABC'})
        self.assertEqual(nonsig['posthoc_status'], 'omnibus_not_significant')
        r = self.run_test(equal_var=False, correction='bonferroni')
        with tempfile.TemporaryDirectory() as d:
            paths = write_report(r, d)
            loaded = json.loads(paths[0].read_text(encoding='utf-8'))
            self.assertEqual(loaded['omnibus'], r['omnibus'])
            self.assertIn('welch_anova', paths[1].read_text(encoding='utf-8'))
            with self.assertRaises(FileExistsError):
                write_report(r, d)
            write_report(r, d, force=True)


if __name__ == '__main__':
    unittest.main()
