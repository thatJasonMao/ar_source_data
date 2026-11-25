"""Small synthetic checks only: no fit, optimizer, real dataset build or training."""
from pathlib import Path
import sys
import unittest
import json
import hashlib
import tempfile
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from shapely.geometry import box

ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT),str(ROOT/'0_RL_FRAMEWORK'),str(ROOT/'0_RL_SOURCE')]
from Build_IRL_Dataset_V3 import temporal_columns, frame_runs, build_trial
from irl_core_v3 import soft_policy
from irl_core import soft_policy as old_soft_policy
from irl_data_v3 import validate_manifest_scope
from gaussian_field_v3 import smooth_display, continuous_display
from unittest.mock import patch


class V3Tests(unittest.TestCase):
    def test_actual_time_history_and_gap(self):
        t=np.array([0,.02,.045,.07,.095,.4,.425,.45])
        frame=pd.DataFrame(dict(timestamp_ns=(t*1e9).astype('int64'),speed=np.arange(8,dtype=float),
            head_rate=np.ones(8),scan_rate=np.ones(8),gaze_depth=np.ones(8)))
        args=SimpleNamespace(max_gap_s=.1,history_s=.06,slow_fraction=.25,slow_duration_s=.4)
        linked=temporal_columns(frame,args)
        self.assertFalse(linked[5]);self.assertTrue(np.isnan(frame.speed_prev.iloc[5]))
        self.assertAlmostEqual(frame.speed_mean_past.iloc[4],2.5)
        self.assertAlmostEqual(frame.acceleration_prev.iloc[3],40.)
        np.testing.assert_array_equal(frame.timestamp_ns,(t*1e9).astype('int64'))

    def test_duration_cap_and_constant_labels(self):
        valid=np.ones(150,bool);action=np.zeros(150,int);linked=np.r_[False,np.ones(149,bool)]
        segment,run,weight,records=frame_runs(valid,action,linked,np.full(150,.02),.02,2.)
        self.assertAlmostEqual(weight.sum()*.02,2.)
        self.assertEqual(len(records),1);self.assertEqual(records[0]['duration_s'],3.)
        self.assertTrue((action==0).all());self.assertTrue((weight>0).all())

    def test_gap_separates_same_action_runs(self):
        linked=np.array([False,True,False,True])
        seg,run,w,records=frame_runs(np.ones(4,bool),np.ones(4,int),linked,np.array([.02,.3,.02,np.nan]),.02,2.)
        np.testing.assert_array_equal(seg,[0,0,1,1]);self.assertEqual(len(records),2)
        self.assertAlmostEqual(sum(r['duration_s'] for r in records),.08)

    def test_full_source_timestamps_and_short_hit(self):
        import Build_IRL_Dataset_V3 as module
        stamps=np.array([0,21,45,68,93,118])*1_000_000+1_700_000_000_000_000_000
        subject=pd.DataFrame({'ParsedTime':pd.to_datetime(stamps,unit='ns'),'PosX':np.arange(6)*.02+1,
            'PosY':1.,'PosZ':1.,'VeloX':1.,'VeloY':0.,'VeloZ':0.})
        row=SimpleNamespace(SubjectFile='unused',ParticipantID='P',Scenario='S',Smoke='HighSmoke',Guidance='Full',Group='A')
        spatial=SimpleNamespace(query=lambda *a:SimpleNamespace(concavity=.1,distance_variation=.2,elongation=2.,horizontal_size=20.,status='ok'))
        decision=SimpleNamespace(query=lambda *a:SimpleNamespace(next_decision_distance=2.,floor='Hall',zone_id='Z',stage='to_gate',status='ok'))
        class Gaze:
            def query_frame_many(self,pid,timestamps,**kw):
                self.timestamps=timestamps;self.kw=kw
                return [SimpleNamespace(is_gazing=i==2,status='ok',sample_age_ms=0.,guidance_types=[]) for i in range(6)]
        gaze=Gaze();args=SimpleNamespace(max_gap_s=.1,history_s=1.,slow_fraction=.25,slow_duration_s=.4,run_cap_s=2.,gaze_match='exact',gaze_max_age_ms=50.)
        with patch.object(module.reference,'read_raw',return_value=(subject,{'raw_rows':6})),patch.object(module,'eye_metrics',return_value=(stamps,np.ones((6,3)),{})):
            df,runs,audit=build_trial(row,'t000','train',args,spatial,decision,gaze)
        np.testing.assert_array_equal(df.timestamp_ns,stamps)
        self.assertEqual(len(df),6);self.assertEqual(int(df.action.sum()),1)
        self.assertEqual(int(df.loc[2,'action']),1);self.assertEqual(int(df.loc[2,'valid']),1)
        self.assertEqual(gaze.kw['match'],'exact')
        self.assertEqual(audit['sampled_rows'],6)

    def test_reject_legacy_dataset(self):
        with self.assertRaisesRegex(ValueError,'full-frame'):
            validate_manifest_scope({'schema':'a5_irl_inputs_v2'})

    def test_irregular_reader_and_no_gap_transition(self):
        from irl_data_v3 import audit_dataset, COHORT_POLICY, NUMERIC
        ns=np.array([0,20,45,400,425])*1_000_000+1_700_000_000_000_000_000
        df=pd.DataFrame({name:np.ones(5) for name in NUMERIC})
        df['timestamp_ns']=ns;df['subject_timestamp_ns']=ns;df['frame_index']=np.arange(5)
        df['dt_next_s']=np.r_[np.diff(ns)/1e9,np.nan];df['run_weight']=1.
        df['action']=[0,1,0,1,0];df['valid']=1;df['after_exit']=0
        df['segment_id']=[0,0,0,1,1];df['transition_valid']=[1,1,0,1,0]
        df['previous_action']=[-1,0,1,-1,1]
        identity=dict(trial_id='t000',participant_id='P',guidance='Full',smoke='HighSmoke',split='train')
        for key,value in identity.items():df[key]=value
        with tempfile.TemporaryDirectory(dir=ROOT/'TEST',prefix='v3_check_') as tmp:
            path=Path(tmp);(path/'trials').mkdir();csv=path/'trials/t000.csv.gz'
            df.to_csv(csv,index=False,compression='gzip')
            record=dict(identity,file='trials/t000.csv.gz',rows=5,sha256=hashlib.sha256(csv.read_bytes()).hexdigest())
            manifest=dict(schema='a5_irl_full_frames_v3',sampling_mode='all_unique_subject_timestamps',status='complete',
                cohort_policy=COHORT_POLICY,cohort_trials=1,cohort_participants=1,excluded_trials=0,
                participant_splits={'P':'train'},config={'step_s':.025,'max_gap_s':.1},trials=[record])
            (path/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            self.assertEqual(audit_dataset(path)['checks'],'passed')
            df.loc[2,'transition_valid']=1;df.to_csv(csv,index=False,compression='gzip')
            with self.assertRaisesRegex(ValueError,'Transition bridges'):
                audit_dataset(path,verify=False)

    def test_duration_discount_reduces_to_v1(self):
        reward=torch.tensor([[0.,1.],[.3,-.3]])
        p=torch.tensor([[[.5,.5],[.2,.8]],[[1.,0.],[0.,1.]]])
        old,_=old_soft_policy(reward,p,4,.98)
        new,_=soft_policy(reward,p,4,.98,mean_dt=torch.full((2,2),.2))
        torch.testing.assert_close(old,new)
        self.assertAlmostEqual(.98**(.02/.2*250),.98**25)

    def test_gaussian_keeps_constant_signed_value_and_raw(self):
        values=np.full((4,4),-2.);values[0,0]=np.nan;before=values.copy()
        coords=(np.arange(4)+.5)*.4
        result=smooth_display(values,coords,coords,box(0,0,1.6,1.6))
        np.testing.assert_array_equal(values,before)
        np.testing.assert_allclose(result[np.isfinite(result)],-2.)
        self.assertTrue(np.isnan(result[0,0]))

    def test_wall_prevents_gaussian_leak(self):
        values=np.array([[0.,0.,10.,10.],[0.,0.,10.,10.]])
        walkable=box(0,0,.79,.8).union(box(.81,0,1.6,.8))
        result=smooth_display(values,np.array([.2,.6,1.,1.4]),np.array([.2,.6]),walkable,sigma_m=1.)
        np.testing.assert_allclose(result[:,:2],0.)
        np.testing.assert_allclose(result[:,2:],10.)

    def test_display_upsampling_only(self):
        values=np.array([[1.,2.],[3.,np.nan]]);raw=values.copy();edges=np.arange(3)*.4
        display,xe,ze=continuous_display(values,edges,edges,box(0,0,.8,.8),4)
        self.assertEqual(display.shape,(8,8));np.testing.assert_array_equal(raw,values)
        self.assertTrue(np.isnan(display[4:,4:]).all())

    def test_display_does_not_reintroduce_wall_mixing(self):
        values=np.array([[0.,0.,10.,10.],[0.,0.,10.,10.]])
        edges=np.arange(5)*.4;ze=np.arange(3)*.4
        walkable=box(0,0,.79,.8).union(box(.81,0,1.6,.8))
        display,xe,_=continuous_display(values,edges,ze,walkable,4)
        xc=(xe[:-1]+xe[1:])/2
        np.testing.assert_allclose(display[:,xc<.79],0.)
        np.testing.assert_allclose(display[:,xc>.81],10.)

    def test_plot_function_exports_raw_before_drawing(self):
        import inspect, Plot_IRL_V3
        source=inspect.getsource(Plot_IRL_V3.plot_run)
        self.assertNotIn('smooth_display(',source)
        self.assertLess(source.index('field.to_csv('),source.index('draw_field('))


if __name__=='__main__':unittest.main(verbosity=2)
