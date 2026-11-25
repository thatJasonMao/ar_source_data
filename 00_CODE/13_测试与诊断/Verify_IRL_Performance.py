"""No fit, optimizer step, real training, or figure generation in these checks."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'0_RL_FRAMEWORK'),str(ROOT/'0_RL_SOURCE'),str(ROOT)]
from irl_performance import (slowdown_trace,prepare_trial,PerformanceEncoder,ConditionalDynamics,
    person_weights,make_reference_bank,cost_to_go,PerformanceIRLQuery,METHOD,EXTRA)
from irl_data import NUMERIC
from irl_core import LinearReward,soft_policy,weighted_nll
from Spatial_IRL_Rule import SPATIAL,reward_mask
from Train_IRL_Performance import parser,validate_args,trigger_plots


def trial():
    return pd.DataFrame(dict(speed=[2.,1.,0.,0.,0.,2.],speed_prev=[2.,2.,1.,0.,0.,0.],
        speed_mean_past=[2.,2.,1.,0.,0.,0.],valid=[1]*6,segment_id=[0]*6,
        timestamp_ns=np.arange(6,dtype=np.int64)*200_000_000,after_exit=[0]*6,
        stage=['to_gate']*6,action=[0,1,1,0,0,0]))


def conditional_records(nfeatures):
    records=[]
    for a in [0,1]:
        coef=np.zeros(nfeatures,np.float32);coef[0]=.2 if a==0 else .05
        records.append(dict(guidance='Full',smoke='HighSmoke',action=a,classes=np.array([a]),
            coef=np.zeros((1,nfeatures),np.float32),intercept=np.zeros(1,np.float32),
            cost_coef=coef,pairs=100,converged=True))
    return records


class PerformanceChecks(unittest.TestCase):
    def test_frozen_cost_and_recovery(self):
        d=trial();t=slowdown_trace(d.speed,d.speed_mean_past,np.ones(6,bool),d.segment_id,.2)
        np.testing.assert_allclose(t['slowdown_cost_s'],[0,0,.2,.2,.2,0])
        self.assertEqual(t['slow_anchor_mps'][4],2.)
        self.assertEqual(t['slow_active_now'][5],0)

    def test_deficit_magnitude_not_just_event(self):
        t=slowdown_trace([2.,1.,1.],[2.,2.,1.5],[True]*3,[0]*3,.2)
        self.assertAlmostEqual(t['slowdown_cost_s'][2],.1)

    def test_cost_independent_of_gaze_and_states_causal(self):
        d=trial();first=prepare_trial(d,.2);d['action']=1-d.action;second=prepare_trial(d,.2)
        np.testing.assert_array_equal(first.slowdown_cost_s,second.slowdown_cost_s)
        short=prepare_trial(d.iloc[:4],.2)
        np.testing.assert_array_equal(first[EXTRA].iloc[:4].to_numpy(),short[EXTRA].to_numpy())
        self.assertAlmostEqual(first._next_cost_s.iloc[1],.2)
        self.assertEqual(first.speed_prev.iloc[1],d.speed.iloc[1])

    def test_gaps_are_censored_and_exit_is_absorbing(self):
        d=trial();d.loc[2,'valid']=0;d.loc[3:,'segment_id']=1
        result=prepare_trial(d,.2)
        self.assertFalse(result._pair_valid.iloc[1])
        self.assertTrue(np.isnan(result._next_cost_s.iloc[1]))
        self.assertEqual(result.slow_active_now.iloc[3],0)
        d=trial();d.loc[5,['stage','after_exit','valid']]=['arrived_exit',1,0]
        result=prepare_trial(d,.2)
        self.assertTrue(result._terminal_next.iloc[4])
        self.assertEqual(result._next_cost_s.iloc[4],0)
        self.assertFalse(result._pair_valid.iloc[5])

    def test_known_penalty_changes_irl_gradient(self):
        p=torch.ones((1,2,1));cost=torch.tensor([[0.,.2]])
        gradients=[]
        for weight in [0.,1.]:
            theta=torch.tensor(0.,requires_grad=True)
            residual=torch.stack([-theta/2,theta/2])[None,:]
            logp,_=soft_policy(residual-weight*cost,p,2)
            gradient=torch.autograd.grad(weighted_nll(logp,torch.tensor([[1.,1.]])),theta)[0]
            gradients.append(gradient.item())
        self.assertAlmostEqual(gradients[0],0.)
        self.assertGreater(abs(gradients[1]),.01)

    def test_cost_to_go_sign_units_and_horizon(self):
        p=np.array([[[1,0],[0,1]],[[0,1],[0,1]]],np.float32)
        c=np.array([[.2,.05],[0,0]],np.float32);pi=np.full((2,2),.5,np.float32)
        q1,_=cost_to_go(pi,p,c,1);q2,future=cost_to_go(pi,p,c,2)
        self.assertAlmostEqual(float(q1[0,0]-q1[0,1]),.15,places=6)
        self.assertAlmostEqual(float(q2[0,0]-q2[0,1]),.275,places=6)
        self.assertEqual(float(q2[1].sum()),0.)
        np.testing.assert_allclose(future,[.125,0])

    def test_conditional_serialization_and_dense_query_agree(self):
        dynamics=ConditionalDynamics(conditional_records(3),2,.2)
        x=np.array([[1,0,0],[1,2,3]],np.float32)
        frame=pd.DataFrame(dict(guidance=['Full']*2,smoke=['HighSmoke']*2))
        p,c=dynamics.dense(x,frame);q,_=dynamics.cost_queries(x,frame,np.array([.4,0]))
        np.testing.assert_allclose(q,c+np.einsum('nak,k->na',p,[.4,0]))
        np.testing.assert_allclose(p.sum(2),1)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.npz';dynamics.save(path);loaded=ConditionalDynamics.load(path)
            np.testing.assert_array_equal(loaded.dense(x,frame)[0],p)

    def test_weighting_and_shared_reference(self):
        base={n:1. for n in NUMERIC[5:]+EXTRA};base['previous_action']=0
        rows=[dict(**base,participant_id='p1',trial_id='one',floor='Hall',stage='to_gate',run_weight=1.)]*20
        rows += [dict(**{**base,'speed_prev':2.},participant_id='p2',trial_id='two',floor='Hall',stage='to_gate',run_weight=1.)]
        frame=pd.DataFrame(rows);w=person_weights(frame)
        self.assertAlmostEqual(w[:20].sum(),w[20])
        bank=make_reference_bank(frame,4)['Hall/to_gate']
        self.assertEqual([r['speed_prev'] for r in bank],[1.,1.,2.,2.])
        self.assertNotIn('trial_id',bank[0]);self.assertNotIn('guidance',bank[0])

    def test_spatial_mask_excludes_current_behavior(self):
        names=PerformanceEncoder().names;mask=reward_mask(names)
        for name in ['scaled_speed_at_action','scaled_acceleration_at_action','slow_active_now','slow_age_over_10s']:
            self.assertEqual(mask[names.index(name)],0)
        self.assertEqual(mask[names.index('scaled_concavity')],1)

    def test_global_query_without_raw_dataset(self):
        encoder=PerformanceEncoder();encoder.median=np.zeros(len(NUMERIC));encoder.scale=np.ones(len(NUMERIC))
        sample={n:1. for n in NUMERIC};sample.update(dict(guidance='Full',smoke='HighSmoke',floor='Hall',
            stage='to_gate',previous_action=0,slow_active_now=0.,slow_deficit_now=0.,slow_age_s=0.,slow_anchor_mps=0.))
        frame=pd.DataFrame([sample]);x=encoder.transform(frame);features=x.shape[1]
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            config=dict(status='complete',method=METHOD,dataset='NONEXISTENT',horizon=3,gamma=.98,slow_cost_weight=1.,min_support=1)
            (run/'config.json').write_text(json.dumps(config))
            (run/'encoder.json').write_text(json.dumps(dict(encoder=encoder.as_dict(),contexts={'Full/HighSmoke/0':[0,1]})))
            reference={k:sample[k] for k in NUMERIC[5:]+['previous_action']+EXTRA}
            (run/'reference_bank.json').write_text(json.dumps({'Hall/to_gate':[reference]}))
            torch.save(LinearReward(features).state_dict(),run/'reward.pt')
            dynamics=ConditionalDynamics(conditional_records(features),2,.2);dynamics.save(run/'conditional_dynamics.npz')
            p,c=dynamics.dense(x,frame)
            np.savez_compressed(run/'mdp.npz',centers=np.vstack([x,np.zeros_like(x)]),
                transitions=np.concatenate([p,np.array([[[0,1],[0,1]]],np.float32)]),cost=np.vstack([c,[[0.,0.]]]).astype(np.float32),
                guidance_index=np.array([3,0]),terminal_index=np.array(1),transition_counts=np.array([[10,10],[0,0]]),radius=np.array([1.,.1]))
            query=PerformanceIRLQuery(run)
            spatial=frame[SPATIAL+['floor','stage','smoke']]
            result=query.query_features(spatial,'Full').iloc[0]
            self.assertGreater(result.performance_benefit_s,0)
            self.assertAlmostEqual(result.performance_benefit_s,result.cost_not_look_s-result.cost_look_s)
            changed=spatial.assign(speed_prev=999.,trial_id='anything')
            np.testing.assert_allclose(query.query_features(changed,'Full').performance_benefit_s,result.performance_benefit_s)

    def test_cli_matches_existing_invocation(self):
        args=parser().parse_args(['--dataset','example','--train','--device','cuda'])
        validate_args(args)
        self.assertEqual(args.horizon,25);self.assertEqual(args.patience,15);self.assertEqual(args.epochs,100)
        self.assertEqual(args.slow_cost_weight,1.);self.assertFalse(args.trajectory_diagnostics)

    def test_trigger_invokes_only_new_plot_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);config=dict(status='complete')
            with patch('Train_IRL_Performance.subprocess.run') as launch:
                trigger_plots(out,config,300)
            args=launch.call_args.args[0]
            self.assertTrue(args[2].endswith('Plot_IRL_Performance.py'))
            self.assertEqual(args[args.index('--run')+1],str(out))
            self.assertEqual(config['plot_status'],'complete')


if __name__=='__main__':unittest.main(verbosity=2)
