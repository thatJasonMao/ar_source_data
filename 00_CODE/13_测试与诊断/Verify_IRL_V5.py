"""Development checks for three-guidance IRL; no experimental fitting."""
from pathlib import Path
import sys,json,tempfile,unittest
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parent.parent/'0_RL_FRAMEWORK'
sys.path.insert(0,str(ROOT))
from Train_IRL_V5 import GUIDANCE,select_by_guidance
from Spatial_IRL_Rule import SpatialIRLRule,SCOPE,reward_mask
from irl_data_v3 import StateEncoder,NUMERIC
from irl_core import LinearReward

class Checks(unittest.TestCase):
    def make_rule(self,path,active=GUIDANCE):
        frame=pd.DataFrame({n:[1.,2.,3.] for n in NUMERIC})
        frame['previous_action']=[0,1,0];frame['floor']='Hall';frame['stage']='to_gate';frame['smoke']='LowSmoke'
        enc=StateEncoder().fit(frame);model=LinearReward(len(enc.names),n_guidance=len(active))
        with torch.no_grad():
            model.shared.copy_(torch.arange(len(enc.names))*.01*torch.tensor(reward_mask(enc.names)))
        path.mkdir(parents=True,exist_ok=True)
        (path/'config.json').write_text(json.dumps(dict(status='complete',reward_scope=SCOPE,active_guidance=active)),encoding='utf-8')
        (path/'encoder.json').write_text(json.dumps(dict(encoder=enc.as_dict())),encoding='utf-8')
        torch.save(model.state_dict(),path/'reward.pt')
        return frame,model,enc

    def test_three_branches_and_spatial_query(self):
        with tempfile.TemporaryDirectory() as temp:
            frame,model,enc=self.make_rule(Path(temp));rule=SpatialIRLRule(temp)
            self.assertEqual(rule.weights.shape[0],3)
            self.assertEqual(rule.guidances,GUIDANCE)
            for i,g in enumerate(GUIDANCE):
                expected=model(torch.tensor(enc.transform(frame)*reward_mask(enc.names)),torch.full((len(frame),),i)).detach().numpy()
                np.testing.assert_allclose(rule.query_features(frame,g),expected[:,1]-expected[:,0],atol=1e-6)
            with self.assertRaises(ValueError):rule.query_features(frame,'Full')
            with self.assertRaises(ValueError):rule.query_position(0,0,0,guidance='Full',smoke='LowSmoke')

    def test_three_guidance_bank_needs_no_mdp(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);bank={}
            for g in GUIDANCE:
                frame,_,_=self.make_rule(path/'Models'/g);bank[g]='Models/'+g
            (path/'config.json').write_text(json.dumps(dict(status='complete',reward_scope=SCOPE,active_guidance=GUIDANCE,model_bank=bank)),encoding='utf-8')
            rule=SpatialIRLRule(path)
            for g in GUIDANCE:self.assertTrue(np.isfinite(rule.query_features(frame,g)).all())
            with self.assertRaises(ValueError):rule.query_features(frame,'Full')

    def test_selection_is_per_guidance_validation_only(self):
        def rec(name,nll,auc,testauc):
            return dict(id=name,status='complete',output=name,test_auc=testauc,validation=dict(by_guidance={g:dict(participant_macro_nll=nll,spatial_reward_auc=auc) for g in GUIDANCE}))
        records={'a':rec('a',.4,.6,.99),'b':rec('b',.404,.7,.1),'c':rec('c',.406,.99,1.)}
        winners,_=select_by_guidance(records)
        self.assertTrue(all(r['id']=='b' for r in winners.values()))
        records['a']['test_auc']=0.;records['b']['test_auc']=1.
        winners,_=select_by_guidance(records)
        self.assertTrue(all(r['id']=='b' for r in winners.values()))

    def test_four_branch_legacy_remains_loadable(self):
        with tempfile.TemporaryDirectory() as temp:
            frame,_,_=self.make_rule(Path(temp),GUIDANCE+['Full'])
            self.assertTrue(np.isfinite(SpatialIRLRule(temp).query_features(frame,'Full')).all())

if __name__=='__main__':unittest.main(verbosity=2)
