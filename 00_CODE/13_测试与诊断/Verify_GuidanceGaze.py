"""Synthetic boundary tests and full-cohort read-only guidance lookup checks."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project-root', type=Path, default=ROOT)
parser.add_argument('--module', type=Path)
parser.add_argument('--report', type=Path)
args = parser.parse_args()
path = args.module or args.project_root/'0_RL_SOURCE'/'Query_GuidanceGaze.py'
spec = importlib.util.spec_from_file_location('guidance_under_test', path)
api = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = api
spec.loader.exec_module(api)


class EventTests(unittest.TestCase):
    def test_threshold_whole_run_and_singletons(self):
        # 5 x 20 ms = exactly 100 ms: first frame is already True offline.
        ns = np.arange(12, dtype=np.int64)*20_000_000
        hits = np.array([1,1,1,1,1,0,1,1,1,1,0,1],bool)
        flags, duration, first, last = api.event_flags(ns,hits)
        np.testing.assert_array_equal(flags,[1,1,1,1,1,0,0,0,0,0,0,0])
        np.testing.assert_array_equal(duration[:5],[100]*5)
        self.assertEqual(first[0],0); self.assertEqual(last[0],4)
        # Single hit with a capped 100 ms weight must still be rejected.
        self.assertFalse(api.event_flags(np.array([0,200_000_000]),np.array([1,0]))[0].any())

    def test_gaps_and_final_sample(self):
        ns = np.array([0,20,40,200,220,240,260,280])*1_000_000
        flags,_,_,_ = api.event_flags(ns,np.ones(8,bool))
        # First event includes a capped final interval; second uses median tail.
        self.assertTrue(flags.all())
        flags = api.event_flags(np.array([0,101,202])*1_000_000,np.ones(3,bool))[0]
        self.assertFalse(flags.any())
        flags = api.event_flags(np.arange(4)*20_000_000,np.ones(4,bool))[0]
        self.assertFalse(flags.any())
        self.assertFalse(api.event_flags(np.array([0]),np.array([1]))[0].any())
        self.assertEqual(len(api.event_flags(np.array([]),np.array([]))[0]),0)

    def test_input_validation(self):
        with self.assertRaises(ValueError): api.event_flags(np.array([1,1]),np.array([1,1]))
        for value in (0,1.2,'NaT','2026-06-04T12:00:00Z','2026-06-04','12:00:00'):
            with self.assertRaises(ValueError): api._timestamp(value)


class QueryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.t0 = pd.Timestamp('2026-06-04 12:00:00')
        stamps = [self.t0+pd.Timedelta(milliseconds=i*20) for i in range(14)]
        self.stamps = [t.strftime(api.TIME_FORMAT) for t in stamps]
        self.filename = 'A5-1_Subject_Simulation_Info_2026_06_04_12_00_00.csv'
        row = dict(Scenario='MiniMap-HighSmoke',Guidance='MiniMap',ParticipantID='P_B1',
                   SubjectFolder='P_B1',FileName=self.filename)
        self.trials = pd.DataFrame([row])
        subject = self.root/'RAW_DATA'/row['Scenario']/'P_B1'/self.filename
        subject.parent.mkdir(parents=True)
        pd.DataFrame({'TimeStamp':self.stamps}).to_csv(subject,index=False)
        self.aoi = self.root/'Results_Guidance_GazeAOI_Integrated'/row['Scenario']/'P_B1'/self.filename.replace('_Subject_Simulation_Info_','_GuidanceGazeIntegrated_')
        self.aoi.parent.mkdir(parents=True)
        self.frame = pd.DataFrame({'TimeStamp':self.stamps,
            'ValidMiniMapAOI':[1]*12+[0,1], 'GazeOnMiniMap':[1]*5+[0]+[1]*4+[0,1,0,0]})
        self.frame.to_csv(self.aoi,index=False)
        self.engine = api.GuidanceGazeQuery(self.root,trials=self.trials)

    def tearDown(self): self.tmp.cleanup()

    def test_lookup_and_batch(self):
        results = self.engine.query_many('P_B1',self.stamps,match='exact')
        self.assertEqual([r.is_gazing for r in results],[True]*5+[False]*7+[None,False])
        self.assertEqual(results[0].event_duration_ms,100)
        self.assertEqual(results[0].guidance_types,('MiniMap',))
        self.assertEqual(results[6].guidance_types,())
        r = self.engine.query('P_B1',self.t0+pd.Timedelta(milliseconds=15))
        self.assertTrue(r.is_gazing); self.assertEqual(r.sample_age_ms,15)
        self.assertEqual(self.engine.query('P_B1',self.t0+pd.Timedelta(milliseconds=15),match='exact').status,'no_sample')
        self.assertEqual(self.engine.query('P_B1',self.t0+pd.Timedelta(milliseconds=15),max_age_ms=10).status,'no_sample')
        self.assertEqual(self.engine.query('unknown',self.stamps[0]).status,'unknown_participant')
        self.assertEqual(self.engine.query('P_B1',self.t0-pd.Timedelta(milliseconds=1)).status,'outside_trial')

    def test_missing_duplicate_and_unsupported(self):
        self.frame = pd.concat([self.frame,self.frame.iloc[[0]]])
        self.frame.to_csv(self.aoi,index=False)
        self.assertEqual(self.engine.query('P_B1',self.stamps[0]).status,'invalid_source')
        self.aoi.unlink()
        self.assertEqual(self.engine.query('P_B1',self.stamps[0]).status,'missing_aoi')
        self.trials['Guidance']='StaticSign'
        engine = api.GuidanceGazeQuery(self.root,trials=self.trials)
        self.assertEqual(engine.query('P_B1',self.stamps[0]).status,'unsupported_guidance')

    def test_ambiguous_trial(self):
        other=self.trials.copy(); other['Scenario']='MiniMap-LowSmoke'
        subject=self.root/'RAW_DATA'/'MiniMap-LowSmoke'/'P_B1'/self.filename
        subject.parent.mkdir(parents=True)
        pd.DataFrame({'TimeStamp':self.stamps}).to_csv(subject,index=False)
        engine=api.GuidanceGazeQuery(self.root,trials=pd.concat([self.trials,other]))
        self.assertEqual(engine.query('P_B1',self.stamps[0]).status,'ambiguous_trial')
        self.assertTrue(engine.query('P_B1',self.stamps[0],scenario='MiniMap-HighSmoke').is_gazing)

    def test_full_switching_and_invalid_break(self):
        self.trials['Guidance']='Full'
        frame=self.frame.copy()
        frame['ValidLightBlanketAOI']=1
        frame['ValidTextAOI']=1
        frame['GazeOnLightBlanket']=[1,1]+[0]*12
        frame['GazeOnMiniMap']=[0,0,1,1]+[0]*10
        frame['GazeOnTextGuidance']=[0]*4+[1]+[0]*9
        frame['AnyGuidanceGaze']=[1]*5+[0]*9
        frame.to_csv(self.aoi,index=False)
        engine=api.GuidanceGazeQuery(self.root,trials=self.trials)
        results=engine.query_many('P_B1',self.stamps[:5],match='exact')
        self.assertTrue(all(r.is_gazing for r in results))
        self.assertEqual([r.guidance_types for r in results],[('LightBlanket',)]*2+[('MiniMap',)]*2+[('Text',)])
        # Invalid middle sample breaks the union event into two subthreshold runs.
        frame.loc[2,['ValidLightBlanketAOI','ValidMiniMapAOI','ValidTextAOI']]=0
        frame.to_csv(self.aoi,index=False)
        engine.clear_cache()
        self.assertEqual([r.is_gazing for r in engine.query_many('P_B1',self.stamps[:5])],[False,False,None,False,False])


def full_cohort():
    # Independent comparison with the actual, unchanged heatmap functions.
    sys.path.insert(0,str(args.project_root))
    import Visualize_Guidance_Information_Acquisition_2D_Heatmaps as heat
    engine=api.GuidanceGazeQuery(args.project_root)
    counts=dict(trials=0,supported_trials=0,unsupported_trials=0,aoi_frames=0,
                retained_frames=0,heatmap_differences=0,singleton_rejections=0,
                exact_threshold_corrections=0,query_checks=0)
    example=None
    for row in engine.trials.itertuples(index=False):
        counts['trials']+=1
        if row.Guidance not in api.CHANNELS:
            bounds=[(lo,hi) for lo,hi,r in engine._participant_index(row.ParticipantID) if r.FileName==row.FileName and r.Scenario==row.Scenario]
            r=engine.query(row.ParticipantID,pd.Timestamp(bounds[0][0]),scenario=row.Scenario)
            assert r.status=='unsupported_guidance',r
            counts['unsupported_trials']+=1
            continue
        data=engine._load_frame(row)
        raw=pd.read_csv(data['path'])
        raw['_time']=pd.to_datetime(raw.TimeStamp,format=api.TIME_FORMAT)
        raw=raw.sort_values('_time')
        valid_columns,gaze=heat.aoi_columns(row.Guidance)
        valid=np.logical_or.reduce([heat.as_bool(raw[c]) for c in valid_columns])
        hits=heat.as_bool(raw[gaze]) & valid
        elapsed,weights=heat.sample_weights(raw['_time'])
        old=heat.retain_guidance_events(hits,elapsed,weights)
        new=data['retained']
        indices=np.flatnonzero(hits)
        breaks=np.flatnonzero((np.diff(indices)>1)|(np.diff(data['ns'][indices])>api.MAX_GAP_NS))+1
        singletons=np.zeros(len(hits),bool)
        for event in np.split(indices,breaks):
            if len(event)==1: singletons[event]=True
        lost=old & ~new; gained=new & ~old
        assert np.all(singletons[lost]), 'Unexpected loss vs heatmap'
        assert np.all(np.isclose(data['duration_ms'][gained],100,atol=1e-6)), 'Unexpected addition vs heatmap'
        counts['supported_trials']+=1
        counts['aoi_frames']+=len(new)
        counts['retained_frames']+=int(new.sum())
        counts['heatmap_differences']+=int((new!=old).sum())
        counts['singleton_rejections']+=int(lost.sum())
        counts['exact_threshold_corrections']+=int(gained.sum())
        # Check actual first/middle/last frames plus a retained event onset.
        chosen={0,len(new)//2,len(new)-1}
        if new.any(): chosen.add(int(np.flatnonzero(new)[0]))
        for i in chosen:
            r=engine.query(row.ParticipantID,pd.Timestamp(int(data['ns'][i])),scenario=row.Scenario,match='exact')
            if r.status=='outside_trial': continue  # AOI may extend beyond Subject recording
            assert r.status==('ok' if valid[i] else 'invalid_aoi'),r
            assert r.is_gazing==(bool(new[i]) if valid[i] else None),r
            counts['query_checks']+=1
            if example is None and r.is_gazing: example=api.asdict(r)
        if counts['supported_trials']%30==0: print(f"Verified {counts['supported_trials']}/180 AOI trials",flush=True)
    assert counts['supported_trials']==180 and counts['unsupported_trials']==90,counts
    counts['example']=example
    return counts


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful(): raise SystemExit(1)
    report=full_cohort()
    report['synthetic_tests']=result.testsRun
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
