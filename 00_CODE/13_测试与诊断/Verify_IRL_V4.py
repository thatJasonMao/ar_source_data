"""V4 synthetic CPU checks; never fit or search the experiment dataset."""
from pathlib import Path
import sys
import unittest
import tempfile
import time
import json
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss,brier_score_loss,roc_auc_score,balanced_accuracy_score

ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT),str(ROOT/'0_RL_FRAMEWORK'),str(ROOT/'0_RL_SOURCE')]
from irl_search_v4 import (write_json,digest,recapped_weights,count_metrics,baseline_reports,
    coarse_configs,explore_configs,cache_key,rank_records,selection_score,summarize_policy,
    save_observations,load_observations,load_cache)
from Train_IRL_V4 import parser,validate_args,devices_for,Search
from IRL_V4_Worker import cpu_tree,train_candidate,configure_device
from irl_data_v3 import StateEncoder,NUMERIC


def toy_cache(folder,dataset):
    from spatial_isovist_core import DEFAULT_GEOMETRY_PATH
    dataset.mkdir();folder.mkdir()
    write_json(dataset/'manifest.json',dict(source_hashes={str(Path(DEFAULT_GEOMETRY_PATH).resolve()):'synthetic_geometry_hash'}))
    (dataset/'action_runs.csv').write_text('trial_id,run_id,duration_s\ntoy,0,1\n')
    frame=pd.DataFrame({name:[.1,.2] for name in NUMERIC})
    frame['floor']='Hall';frame['stage']='to_gate';frame['smoke']='LowSmoke';frame['previous_action']=[0,1]
    encoder=StateEncoder().fit(frame);centers=np.vstack([encoder.transform(frame),np.zeros((1,len(encoder.names)),np.float32)])
    transition=np.array([[[.8,.2,0],[.3,.7,0]],[[.7,.3,0],[.2,.8,0]],[[0,0,1],[0,0,1]]],np.float32)
    counts=np.array([[.35,.15],[.1,.4],[0,0]],np.float32)
    np.savez_compressed(folder/'mdp.npz',centers=centers,transitions=transition,counts=counts,
        raw_counts=counts*100,transition_counts=np.ones((3,2)),guidance_index=np.array([3,3,0]),
        radius=np.ones(3),terminal_index=2,mean_dt=np.full((3,2),.022,np.float32))
    write_json(folder/'encoder.json',dict(encoder=encoder.as_dict(),contexts={'Full/LowSmoke/0':(0,1),'Full/LowSmoke/1':(1,1)},step_s=.022))
    observations={}
    for split in ['train','val']:
        natural=np.array([[[7.,3.],[2.,8.],[0.,0.]]]);contexts=np.array([[[9.,11.]]])
        observations[split]=dict(natural=natural,capped=natural.copy(),balanced=natural/natural.sum(),
            contexts=contexts,context_capped=contexts.copy(),context_balanced=contexts/contexts.sum(),people=[split+'_person'])
    save_observations(folder/'observations.npz',observations,['Full/LowSmoke/to_gate'])
    write_json(folder/'metadata.json',dict(status='complete',train_rows=20,
        manifest_sha256=digest(dataset/'manifest.json'),runs_sha256=digest(dataset/'action_runs.csv'),
        files_sha256={name:digest(folder/name) for name in ['mdp.npz','encoder.json','observations.npz']}))
    return observations


class V4Tests(unittest.TestCase):
    def test_recap_preserves_labels_and_changes_duration_mass(self):
        frame=pd.DataFrame(dict(run_id=[0,0,1,-1],valid=[1,1,1,0],action=[0,0,1,-1],run_weight=[.1,.1,1.,0.]))
        before=frame.action.copy()
        w=recapped_weights(frame,5.,2.,{0:20.,1:1.})
        np.testing.assert_allclose(w,[.25,.25,1.,0.]);pd.testing.assert_series_equal(before,frame.action)
        np.testing.assert_array_equal(recapped_weights(frame,2.,2.,{0:20.,1:1.}),frame.run_weight)

    def test_count_metrics_equal_full_frame_metrics(self):
        counts=np.array([[7,3],[2,8],[5,0]]);p=np.array([.2,.7,.1]);ys=[];ps=[]
        for row,prob in zip(counts,p):
            ys += [0]*row[0]+[1]*row[1];ps += [prob]*row.sum()
        m=count_metrics(counts,p)
        self.assertAlmostEqual(m['nll'],log_loss(ys,ps));self.assertAlmostEqual(m['brier'],brier_score_loss(ys,ps))
        self.assertAlmostEqual(m['auc'],roc_auc_score(ys,ps));self.assertAlmostEqual(m['balanced_accuracy'],balanced_accuracy_score(ys,np.array(ps)>=.5))

    def test_person_macro_does_not_weight_long_person_more(self):
        natural=np.array([[[90.,10.],[0,0]],[[0,0],[1.,9.]]])
        obs=dict(natural=natural,capped=natural,balanced=natural)
        m=summarize_policy(obs,np.array([[.8,.2],[.3,.7]]),np.array([3,3]))
        expected=np.mean([count_metrics(c,np.array([.2,.7]))['nll'] for c in natural])
        self.assertAlmostEqual(m['participant_macro_nll'],expected)

    def test_baselines_only_fit_training_counts(self):
        train=dict(contexts=np.array([[[8.,2.]]]),context_balanced=np.array([[[.8,.2]]]))
        evaluation=dict(contexts=np.array([[[0.,10.]]]),context_balanced=np.array([[[0.,1.]]]))
        result=baseline_reports(train,evaluation,['Full/LowSmoke/to_gate'])
        expected=-np.log(2.5/11.)
        self.assertAlmostEqual(result['guidance_natural']['nll'],expected)

    def test_search_plan_has_baseline_and_unique_configs(self):
        args=parser().parse_args(['--dataset','unused']);configs=coarse_configs(args)
        self.assertEqual(len(configs),12);self.assertEqual(configs[0]['lr'],.01);self.assertEqual(configs[0]['l2'],.01)
        self.assertEqual(len({json.dumps(c,sort_keys=True) for c in configs}),12)
        extra=explore_configs(configs[:2]);self.assertEqual(len(extra),18)
        self.assertEqual({c['clusters'] for c in extra},{16,24,48})

    def test_cache_key_ignores_optimizer_but_tracks_state(self):
        args=parser().parse_args(['--dataset','unused']);c=coarse_configs(args)[0]
        key=cache_key(c,'data','runs')
        self.assertEqual(key,cache_key(dict(c,lr=.03,l2=.1),'data','runs'))
        self.assertNotEqual(key,cache_key(dict(c,run_cap_s=5.),'data','runs'))
        self.assertNotEqual(key,cache_key(dict(c,seed=c['seed']+1),'data','runs'))

    def test_argument_budget_and_multi_gpu(self):
        args=parser().parse_args(['--dataset','unused','--devices','cuda:0,cuda:1','--jobs','2'])
        validate_args(args);self.assertEqual(devices_for(args),['cuda:0','cuda:1'])
        args.reserve_minutes=200
        with self.assertRaises(ValueError):validate_args(args)

    def test_selection_is_validation_nll_not_test_auc(self):
        records=[dict(id='a',status='complete',score=.6,validation={'brier':.2},test_auc=.99),
                 dict(id='b',status='complete',score=.5,validation={'brier':.3},test_auc=.51)]
        self.assertEqual(rank_records(records)[0]['id'],'b')
        with self.assertRaises(ValueError):selection_score({'participant_macro_nll':None},'participant_macro_nll')

    def test_cpu_checkpoint_is_a_snapshot(self):
        t=torch.tensor([1.]);copy=cpu_tree({'x':t});t[0]=2.
        self.assertEqual(copy['x'].item(),1.)

    def test_default_cuda_has_explicit_device_index(self):
        with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.set_device') as set_device:
            self.assertEqual(configure_device('cuda').type,'cuda');set_device.assert_called_with(0)
            configure_device('cuda:1');set_device.assert_called_with(1)

    def test_cache_checksum_and_train_val_only(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'TEST',prefix='v4_check_') as tmp:
            root=Path(tmp);toy_cache(root/'cache',root/'dataset')
            mdp,observations,keys=load_cache(root/'cache',root/'dataset')
            self.assertEqual(set(observations),{'train','val'})
            self.assertEqual(mdp['terminal_index'],2)
            (root/'cache/encoder.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'checksum'):load_cache(root/'cache',root/'dataset')

    def test_synthetic_worker_train_resume_and_no_test(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'TEST',prefix='v4_check_') as tmp:
            root=Path(tmp);toy_cache(root/'cache',root/'dataset')
            args=parser().parse_args(['--dataset','unused']);c=dict(coarse_configs(args)[0],horizon_seconds=.044)
            job=dict(output=str(root/'candidate'),dataset=str(root/'dataset'),cache=str(root/'cache'),id='toy',config=c,
                device='cpu',threads=1,epochs=4,resume=False,lr_patience=1,min_delta=1e-6,min_epochs=1,
                patience=4,deadline=time.time()+60,selection_metric='participant_macro_nll')
            result=train_candidate(job)
            self.assertEqual(result['status'],'complete');self.assertEqual(result['epochs_run'],4)
            self.assertNotIn('test',result)
            saved=torch.load(root/'candidate/checkpoint.pt',weights_only=True)
            reward=torch.load(root/'candidate/reward.pt',weights_only=True)
            for k in reward:torch.testing.assert_close(reward[k],saved['best_state'][k])
            job.update(resume=True,epochs=6);result=train_candidate(job)
            self.assertEqual(result['epochs_run'],6)
            self.assertEqual(len(pd.read_csv(root/'candidate/training_history.csv')),6)

    def test_stop_marker_retains_promoted_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'TEST',prefix='v4_check_') as tmp:
            root=Path(tmp);toy_cache(root/'cache',root/'dataset')
            c=dict(coarse_configs(parser().parse_args(['--dataset','unused']))[0],horizon_seconds=.044)
            job=dict(output=str(root/'candidate'),dataset=str(root/'dataset'),cache=str(root/'cache'),id='toy',config=c,
                device='cpu',threads=1,epochs=1,resume=False,lr_patience=1,min_delta=1e-6,min_epochs=1,
                patience=4,deadline=time.time()+60,selection_metric='participant_macro_nll')
            result=train_candidate(job);(root/'candidate/STOP').touch();job.update(resume=True,epochs=4)
            stopped=train_candidate(job)
            self.assertEqual(stopped['reason'],'phase_budget');self.assertEqual(stopped['epochs_run'],1)

    def test_two_real_cpu_worker_processes_and_hard_trial_cap(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'TEST',prefix='v4_check_') as tmp:
            root=Path(tmp);toy_cache(root/'cache',root/'dataset');(root/'search').mkdir()
            args=parser().parse_args(['--dataset',str(root/'dataset'),'--device','cpu','--jobs','2',
                '--max-trials','2','--screen-epochs','2','--epochs','4','--min-epochs','1'])
            search=Search(args,root/'search',time.time()+45)
            search.cache_for=lambda c:root/'cache'
            search.cache_status[str(root/'cache')]='ready'
            configs=[dict(c,horizon_seconds=.044) for c in coarse_configs(args)[:3]]
            try:search.phase(configs,'synthetic_parallel',2,time.time()+40)
            finally:search.stop()
            self.assertEqual(len(search.records),2)
            self.assertEqual(len(search.ranked()),2)
            self.assertFalse(search.active)
            self.assertTrue((root/'search/search_summary.csv').exists())


if __name__=='__main__':unittest.main(verbosity=2)
