"""No-training checks for A5 IRL math, causal input utilities, and query IO.

No optimizer steps or clustering fits are executed. Synthetic numerical
gradients verify the objective while leaving all parameters unchanged.
"""
from pathlib import Path
import json
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT),str(ROOT/'0_RL_SOURCE'),str(ROOT/'0_RL_FRAMEWORK')]
from Build_IRL_Dataset import ns_array,previous_indices,segments_and_runs,slowdown_states,split_roster
from irl_core import LinearReward,soft_policy,weighted_nll,expected_burden
from irl_data import StateEncoder,NUMERIC,GUIDANCE,assign_states,audit_dataset,iter_trials
from IRL_Cohort import SCHEMA,COHORT_POLICY,select_irl_trials
from common import load_trials
from Query_IRL import IRLQuery


class DataContracts(unittest.TestCase):
    def test_pandas_time_units(self):
        t=pd.DatetimeIndex(['2026-09-26T12:00:00.000','2026-09-26T12:00:00.200']).as_unit('us')
        ns=ns_array(t)
        self.assertEqual(ns[1]-ns[0],200_000_000)
        i,v,age=previous_indices(ns,np.array([ns[0]-1,ns[0]+25_000_000,ns[0]+51_000_000]))
        self.assertEqual(v.tolist(),[False,True,False]);self.assertAlmostEqual(age[1],.025)

    def test_runs_gaps_and_long_constant_labels(self):
        a=np.array([0]*30+[-1]+[1,0,0]);v=a>=0
        seg,run,w,r=segments_and_runs(v,a,10)
        self.assertEqual(a.tolist(),[0]*30+[-1]+[1,0,0])
        self.assertAlmostEqual(w[:30].sum(),10)
        self.assertEqual(seg[30],-1);self.assertNotEqual(seg[29],seg[31])
        self.assertEqual([x['steps'] for x in r],[30,1,2])

    def test_persistent_stop_and_recovery(self):
        speed=np.array([2,2,.5,0,0,0,0,2,np.nan,0])
        baseline=np.array([np.nan,2,2,1,.1,0,0,0,2,0])
        slow=slowdown_states(speed,baseline,.25,2)
        self.assertEqual(slow.tolist(),[0,0,0,1,1,1,1,0,0,0])
        # A future observation cannot alter any earlier event flag.
        np.testing.assert_array_equal(slow[:5],slowdown_states(speed[:5],baseline[:5],.25,2))

    def test_participant_split(self):
        roster=pd.DataFrame([dict(ParticipantID=f'{g}{i}',Group=g) for g in 'ABC' for i in range(15) for trial in range(6)])
        splits=split_roster(roster,1)
        self.assertEqual(len(splits),45)
        self.assertEqual(list(splits.values()).count('train'),27)
        self.assertEqual(list(splits.values()).count('val'),9)

    def test_confirmed_cohort_and_stable_person_splits(self):
        canonical=load_trials()
        original=canonical.copy(deep=True)
        included,excluded=select_irl_trials(canonical)
        self.assertEqual((len(canonical),len(included),len(excluded)),(270,178,92))
        self.assertEqual(included.Guidance.value_counts().to_dict(),
                         {'Full':90,'ArrowRibbon':30,'MiniMap':30,'TextBar':28})
        self.assertEqual(included.ParticipantID.nunique(),45)
        self.assertEqual(set(included.loc[included.ParticipantID.eq('LJN_C12'),'Guidance']),{'Full'})
        self.assertEqual(set(excluded.loc[excluded.Guidance.eq('TextBar'),'trial_id']),{'t246','t261'})
        self.assertEqual(split_roster(canonical,20260926),split_roster(included,20260926))
        pd.testing.assert_frame_equal(canonical,original)
        reversed_included,_=select_irl_trials(canonical.iloc[::-1])
        pd.testing.assert_frame_equal(included,reversed_included)

    def test_old_cohort_cannot_be_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'manifest.json').write_text(json.dumps(dict(schema='a5_irl_inputs_v1',status='complete')))
            with self.assertRaisesRegex(ValueError,'rebuild older inputs'):
                list(iter_trials(tmp))

    def test_exclusions_checked_before_split_filtering(self):
        for pid,guidance in [('A1','StaticSign'),('LJN_C12','TextBar')]:
            with self.subTest(pid=pid,guidance=guidance),tempfile.TemporaryDirectory() as tmp:
                manifest=dict(schema=SCHEMA,status='complete',cohort_policy=COHORT_POLICY,cohort_trials=1,
                    trials=[dict(trial_id='excluded',participant_id=pid,guidance=guidance,split='test')])
                (Path(tmp)/'manifest.json').write_text(json.dumps(manifest))
                # Requesting train must still reject an excluded test trial.
                with self.assertRaisesRegex(ValueError,'Excluded IRL trial'):
                    list(iter_trials(tmp,splits=['train']))

    def test_trial_payload_cannot_hide_excluded_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            record=dict(trial_id='t000',participant_id='A1',guidance='TextBar',smoke='LowSmoke',
                        split='train',file='trial.csv')
            pd.DataFrame([{**record,'participant_id':'LJN_C12'}]).to_csv(Path(tmp)/'trial.csv',index=False)
            manifest=dict(schema=SCHEMA,status='complete',cohort_policy=COHORT_POLICY,cohort_trials=1,trials=[record])
            (Path(tmp)/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'Trial identity differs'):
                list(iter_trials(tmp))


class IRLMath(unittest.TestCase):
    def test_absorbing_completion_without_entropy_bonus(self):
        p=torch.tensor([[[1.,0.],[0.,1.]],[[0.,1.],[0.,1.]]])
        r=torch.zeros((2,2))
        logp,q=soft_policy(r,p,10,terminal_index=1)
        torch.testing.assert_close(q,torch.zeros_like(q))
        policy=torch.tensor([[0.,1.],[.5,.5]])
        costs=torch.tensor([[1.,1.],[0.,0.]])
        _,burden=expected_burden(policy,p,costs,10,.2)
        self.assertAlmostEqual(burden[0].item(),.2)
        self.assertEqual(burden[1].item(),0)

    def test_policy_depends_on_future(self):
        # Choosing 1 from state0 reaches state1, where either action has reward2.
        p=torch.tensor([[[1.,0.],[0.,1.]],[[0.,1.],[0.,1.]]],dtype=torch.float64)
        r=torch.tensor([[0.,0.],[2.,2.]],dtype=torch.float64)
        lp,_=soft_policy(r,p,horizon=1);self.assertAlmostEqual(lp.exp()[0,1].item(),.5)
        lp,_=soft_policy(r,p,horizon=2);self.assertGreater(lp.exp()[0,1].item(),.8)

    def test_gradient_without_training(self):
        p=torch.tensor([[[.8,.2],[.2,.8]],[[.3,.7],[.1,.9]]],dtype=torch.float64)
        features=torch.tensor([[1.,.5],[1.,-.7]],dtype=torch.float64)
        g=torch.tensor([0,1]);model=LinearReward(2).double()
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        counts=torch.tensor([[2.,1.],[1.,3.]],dtype=torch.float64)
        lp,_=soft_policy(model(features,g),p,4)
        loss=weighted_nll(lp,counts);grad=torch.autograd.grad(loss,model.shared)[0]
        self.assertTrue(torch.isfinite(grad).all())
        eps=1e-6
        def f(v):
            contrast=features@v
            r=torch.stack([-contrast/2,contrast/2],1)
            return weighted_nll(soft_policy(r,p,4)[0],counts).item()
        numeric=(f(torch.tensor([eps,0.],dtype=torch.float64))-f(torch.tensor([-eps,0.],dtype=torch.float64)))/(2*eps)
        self.assertAlmostEqual(grad[0].item(),numeric,places=6)
        for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k])

    def test_cost_is_separate_and_bounded(self):
        policy=torch.tensor([[.25,.75]])
        p=torch.ones((1,2,1));c=torch.tensor([[0.,1.]])
        immediate,burden=expected_burden(policy,p,c,25,.2)
        self.assertAlmostEqual(immediate.item(),.75)
        self.assertAlmostEqual(burden.item(),3.75,places=5)

    def test_query_roundtrip_without_fit(self):
        frame=pd.DataFrame([{**{n:1. for n in NUMERIC},'guidance':'Full','smoke':'LowSmoke',
                             'previous_action':0,'floor':'Hall','stage':'to_gate'}])
        encoder=StateEncoder();encoder.median=np.zeros(len(NUMERIC));encoder.scale=np.ones(len(NUMERIC))
        x=encoder.transform(frame)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'config.json').write_text(json.dumps(dict(status='complete',horizon=3,gamma=.98,min_support=5)))
            (root/'encoder.json').write_text(json.dumps(dict(encoder=encoder.as_dict(),contexts={'Full/LowSmoke/0':[0,1]},step_s=.2)))
            np.savez_compressed(root/'mdp.npz',centers=x,transitions=np.ones((1,2,1),np.float32),
                guidance_index=np.array([3]),event_cost=np.array([[0,.5]],np.float32),
                transition_counts=np.array([[10,10]]),radius=np.array([1.]))
            torch.save(LinearReward(x.shape[1]).state_dict(),root/'reward.pt')
            result=IRLQuery(root).query_frame(frame).iloc[0]
            self.assertEqual(result.status,'ok');self.assertAlmostEqual(result.p_look,.5)
            self.assertAlmostEqual(result.trap_probability,.25)
            self.assertAlmostEqual(result.reward_contrast,0.)
            frame.loc[0,'guidance']='StaticSign'
            self.assertEqual(IRLQuery(root).query_frame(frame).iloc[0].status,'unsupported_context')


if __name__=='__main__':unittest.main(verbosity=2)
