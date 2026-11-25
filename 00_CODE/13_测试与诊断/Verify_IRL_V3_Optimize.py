from pathlib import Path
import sys,tempfile,unittest,json
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'0_RL_FRAMEWORK'),str(ROOT/'0_RL_SOURCE')]
from spatial_reward_basis import SpatialRewardBasis
from irl_frame_optimization import state_basis_means,frame_logit
from irl_core_v3 import LinearReward,soft_policy
from Spatial_IRL_Rule import SpatialIRLRule,reward_mask,SCOPE
from irl_data_v3 import StateEncoder,NUMERIC
from irl_search_v4 import write_json

class Checks(unittest.TestCase):
    def inputs(self):
        rng=np.random.default_rng(1)
        names=['bias']+['scaled_'+n for n in NUMERIC[:5]]+['floor_Hall','floor_Platform','stage_to_gate','high_smoke']
        x=rng.normal(size=(100,len(names))).astype(np.float32);x[:,0]=1
        return x,names
    def test_basis_roundtrip_and_finite(self):
        x,n=self.inputs()
        for kind in ['linear','spline','interaction','rbf32','rbf64']:
            b=SpatialRewardBasis.fit(x,n,kind);c=SpatialRewardBasis(json.loads(json.dumps(b.spec)))
            np.testing.assert_array_equal(b.transform(x),c.transform(x));self.assertTrue(np.isfinite(b.transform(x)).all())
            self.assertFalse(any('speed' in k or 'previous' in k for k in b.names))
    def test_participant_bootstrap_matches_explicit_resampling(self):
        from Analyze_IRL_V3_Optimization import participant_bootstrap
        from sklearn.metrics import roc_auc_score
        y=np.array([0,1,0,1,1,0]);scores=np.array([.1,.6,.6,.5,.9,.2]);person=np.array([0,0,1,1,2,2])
        result=participant_bootstrap(y,scores,person,repeats=100)
        rng=np.random.default_rng(20260926);values=[]
        for _ in range(100):
            people=rng.integers(0,3,3);indices=np.concatenate([np.flatnonzero(person==i) for i in people])
            values.append(roc_auc_score(y[indices],scores[indices]))
        np.testing.assert_allclose(result['auc_ci95'],np.quantile(values,[.025,.975]))
    def test_training_knots_do_not_refit_at_query(self):
        x,n=self.inputs();b=SpatialRewardBasis.fit(x,n,'spline');before=json.dumps(b.spec)
        b.transform(x*2);self.assertEqual(before,json.dumps(b.spec))
    def test_guidance_bank_queries_are_independent_of_trials_and_mdp(self):
        from irl_data_v3 import GUIDANCE
        f=pd.DataFrame({n:[.1,.2,.5] for n in NUMERIC});f['floor']='Hall';f['stage']='to_gate';f['smoke']='LowSmoke';f['previous_action']=0
        e=StateEncoder().fit(f);mask=reward_mask(e.names).astype(bool)
        with tempfile.TemporaryDirectory(dir=ROOT/'TEST') as directory:
            root=Path(directory);bank={}
            for i,g in enumerate(GUIDANCE):
                child=root/g;child.mkdir();bank[g]=g
                b=SpatialRewardBasis.fit(e.transform(f)[:,mask],[n for n,a in zip(e.names,mask) if a],'spline' if i%2 else 'linear')
                m=LinearReward(len(b.names));m.shared.data[:]=.03*(i+1)
                write_json(child/'config.json',dict(status='complete',reward_scope=SCOPE));write_json(child/'encoder.json',dict(encoder=e.as_dict()));write_json(child/'reward_basis.json',b.spec);torch.save(m.state_dict(),child/'reward.pt')
            write_json(root/'config.json',dict(status='complete',reward_scope=SCOPE,model_bank=bank));rule=SpatialIRLRule(root)
            self.assertFalse(list(root.rglob('mdp.npz')))
            for g in GUIDANCE:np.testing.assert_allclose(rule.query_features(f,g),SpatialIRLRule(root/g).query_features(f,g))
    def test_state_assignment_avoids_float32_cancellation(self):
        from irl_state_distance import stable_nearest
        x=np.array([[8,8,8],[7.999,8,8]],np.float32)
        centers=np.array([[7.999,8,8],[8,8,8]],np.float32)
        a,_=stable_nearest(x,centers)
        np.testing.assert_array_equal(a,[1,0])
        np.testing.assert_array_equal(a,np.concatenate([stable_nearest(row[None],centers)[0] for row in x]))
    def test_weighted_state_mean(self):
        x=np.array([[1.,2.],[3.,4.],[5.,6.]],np.float32)
        np.testing.assert_allclose(state_basis_means(x,np.array([0,0,1]),np.array([1.,3.,2.]),3),[[2.5,3.5],[5,6],[0,0]])
    def test_frame_likelihood_equals_mdp_at_state_means(self):
        torch.manual_seed(1);m=LinearReward(2);m.shared.data[:]=torch.tensor([.2,.4])
        mean=torch.tensor([[1.,.2],[1.,-.3],[0.,0.]]);g=torch.tensor([3,3,0]);t=torch.tensor([[[.8,.2,0],[.3,.7,0]],[[.7,.3,0],[.2,.8,0]],[[0,0,1],[0,0,1]]]);dt=torch.full((3,2),.02)
        mdp=dict(guide_tensor=g,terminal_index=2)
        z,_=frame_logit(m,mean,g,torch.arange(3),mean,mdp,t,dt,10,.98)
        lp,_=soft_policy(m(mean,g),t,10,.98,2,mean_dt=dt)
        torch.testing.assert_close(torch.sigmoid(z),lp.exp()[:,1]);z.sum().backward();self.assertTrue(torch.isfinite(m.shared.grad).all())
    def test_json_retries_transient_windows_error(self):
        import irl_search_v4
        actual=irl_search_v4.os.replace;calls=[]
        def flaky(a,b):
            calls.append(1)
            if len(calls)<3:raise PermissionError('temporary sharing')
            return actual(a,b)
        with tempfile.TemporaryDirectory(dir=ROOT/'TEST') as d:
            with patch.object(irl_search_v4.os,'replace',side_effect=flaky),patch.object(irl_search_v4.time,'sleep'):
                write_json(Path(d)/'x.json',dict(value=1))
            self.assertEqual(json.loads((Path(d)/'x.json').read_text()),dict(value=1));self.assertEqual(len(calls),3);self.assertFalse(list(Path(d).glob('*.tmp')))
    def test_spatial_query_ignores_human_features_and_matches_model(self):
        f=pd.DataFrame({n:[.1,.2,.5] for n in NUMERIC});f['floor']='Hall';f['stage']='to_gate';f['smoke']='LowSmoke';f['previous_action']=0
        enc=StateEncoder().fit(f);mask=reward_mask(enc.names).astype(bool)
        b=SpatialRewardBasis.fit(enc.transform(f)[:,mask],[n for n,a in zip(enc.names,mask) if a],'interaction');m=LinearReward(len(b.names));m.shared.data[:]=.03
        with tempfile.TemporaryDirectory(dir=ROOT/'TEST') as d:
            p=Path(d);write_json(p/'config.json',dict(status='complete',reward_scope=SCOPE));write_json(p/'encoder.json',dict(encoder=enc.as_dict()));write_json(p/'reward_basis.json',b.spec);torch.save(m.state_dict(),p/'reward.pt')
            rule=SpatialIRLRule(p);a=rule.query_features(f,'Full');f['speed_prev']=100;f['previous_action']=1
            np.testing.assert_allclose(a,rule.query_features(f,'Full'))
            with torch.no_grad():r=m(torch.as_tensor(b.transform(enc.transform(f)[:,mask])),torch.full((3,),3));np.testing.assert_allclose(a,(r[:,1]-r[:,0]).numpy(),rtol=1e-5,atol=1e-5)

if __name__=='__main__':unittest.main(verbosity=2)
