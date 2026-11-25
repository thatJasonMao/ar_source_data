"""Regression tests for canonical TIMELINE identity and cache contracts."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import TimelineTrialInput as ti


def fixture():
    records=[]
    for group,mode in [('A','ArrowRibbon'),('B','MiniMap'),('C','TextBar')]:
        for i in range(1,16):
            person=f'P_{group}{i}'
            for guidance in ['StaticSign',mode,'Full']:
                for smoke in ['HighSmoke','LowSmoke']:
                    scenario=f'{guidance}-{smoke}'
                    records.append(dict(Scenario=scenario,ParticipantID=person,
                        FileName='session.csv',Group=group,SubjectFile=f'/data/{scenario}/{person}/session.csv',EyeGazeFile='eye.csv'))
    return pd.DataFrame(records)


class TrialContractTests(unittest.TestCase):
    def setUp(self):
        self.mock=patch.object(ti.common,'load_trials',return_value=fixture())
        self.mock.start();self.addCleanup(self.mock.stop)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'timeline.csv'
        self.rows=[{k:r[k] for k in (*ti.KEYS,'group')}|{'leave_spawn_ts':'Null'} for r in ti.canonical_trials()]

    def write(self,rows):
        with self.path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=[*ti.KEYS,'group','leave_spawn_ts']);w.writeheader();w.writerows(rows)

    def test_reordering_preserves_identity_and_missing_period(self):
        self.write(list(reversed(self.rows)))
        actual=ti.load_timeline(self.path)
        self.assertEqual([ti.trial_key(r) for r in actual],[ti.trial_key(r) for r in self.rows])
        self.assertEqual(actual[0]['leave_spawn_ts'],'Null')
        self.assertEqual(ti.cohort_metadata(actual)['cohort_trials'],270)

    def test_legacy_extra_missing_duplicate_and_equal_size_replacement_rejected(self):
        foreign=dict(self.rows[0],src_file='superseded.csv')
        for rows in [self.rows+[foreign],self.rows[:-1],self.rows+[self.rows[0]],self.rows[:-1]+[foreign]]:
            with self.subTest(n=len(rows)):
                self.write(rows)
                with self.assertRaises(ValueError):ti.load_timeline(self.path)

    def test_group_and_root_mismatch_rejected(self):
        self.write([dict(self.rows[0],group='X')]+self.rows[1:])
        with self.assertRaises(ValueError):ti.load_timeline(self.path)
        with self.assertRaises(ValueError):ti.canonical_trials('/different/raw/root')

    def test_cache_rejects_wrong_id_foreign_key_and_group(self):
        table=pd.DataFrame([dict(self.rows[0],trial_id=1)])
        ti.validate_trial_table(table,self.rows)
        for col,value in [('trial_id',2),('trial_id',1.5),('subject','TEST'),('group','X')]:
            bad=table.copy();bad[col]=value
            with self.subTest(col=col,value=value),self.assertRaises(ValueError):ti.validate_trial_table(bad,self.rows)

    def test_partial_period_support_valid_but_old_metadata_invalid(self):
        ti.validate_trial_table(pd.DataFrame([dict(self.rows[5],trial_id=6)]),self.rows)
        meta=ti.cohort_metadata(self.rows);ti.validate_metadata(meta,self.rows)
        for bad in [{},dict(meta,cohort_trials=292),dict(meta,cohort_sha256='old')]:
            with self.assertRaises(ValueError):ti.validate_metadata(bad,self.rows)

    def test_legacy_frame_directory_cannot_silently_retain_excluded_trials(self):
        frames=Path(self.temp.name)/'frames';frames.mkdir();(frames/'legacy.csv.gz').write_bytes(b'old')
        with self.assertRaises(ValueError):ti.require_fresh_frame_cohort(self.temp.name,self.rows)
        (Path(self.temp.name)/'run_summary.json').write_text(json.dumps(ti.cohort_metadata(self.rows)))
        ti.require_fresh_frame_cohort(self.temp.name,self.rows)


if __name__=='__main__':unittest.main()
