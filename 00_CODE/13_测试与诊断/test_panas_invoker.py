"""Identity and pairing contract checks for the PANAS invoker."""
import importlib.util
from pathlib import Path
import unittest

import pandas as pd

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / 'Invoker_PANAS.py' if (HERE / 'Invoker_PANAS.py').exists() else HERE.parent / 'Invoker_PANAS.py'
spec = importlib.util.spec_from_file_location('panas_contract', MODULE_PATH)
invoker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(invoker)


class PairingContract(unittest.TestCase):
    def setUp(self):
        self.trials = pd.DataFrame({'ParticipantID': ['S1']*6+['S2']*6+['S3']*6+['S4']*6,
                                    'Group': ['A']*24})
        self.raw = pd.DataFrame({'ParticipantID': ['S3','S1','S4','S2'], 'Group': ['A']*4,
             'IncludedTrialCount': [6]*4, 'MeasurementLevel': ['participant_experiment_pre_post']*4,
             'PA_Pre': [10,12,14,16], 'PA_Post': [11,14,None,15],
             'NA_Pre': [9,10,11,12], 'NA_Post': [10,11,None,13]})

    def test_sort_and_complete_case_pairing(self):
        paired, audit = invoker.prepare_data(self.raw, self.trials)
        self.assertEqual(paired.ParticipantID.tolist(), ['S1','S2','S3'])
        self.assertEqual(paired.PA_Delta_PostMinusPre.tolist(), [2,-1,1])
        self.assertEqual((audit.Status == 'excluded_missing_measurement').sum(), 1)

    def test_duplicate_and_unknown_ids_rejected(self):
        bad = self.raw.copy()
        bad.loc[0,'ParticipantID'] = 'S1'
        with self.assertRaises(ValueError):
            invoker.prepare_data(bad, self.trials)
        bad.loc[0,'ParticipantID'] = 'UNKNOWN'
        with self.assertRaises(ValueError):
            invoker.prepare_data(bad, self.trials)

    def test_invalid_score_not_silently_omitted(self):
        bad = self.raw.copy()
        bad.loc[0,'PA_Pre'] = 46
        with self.assertRaises(ValueError):
            invoker.prepare_data(bad, self.trials)


if __name__ == '__main__':
    unittest.main()
