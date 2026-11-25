"""Spatial reward/query/trigger checks. No training and no actual map generation."""
from pathlib import Path
import json,sys,tempfile,unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from shapely.geometry import box,Polygon
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'0_RL_FRAMEWORK'),str(ROOT/'0_RL_SOURCE'),str(ROOT)]
from Spatial_IRL_Rule import SpatialIRLRule,reward_mask,SPATIAL,SCOPE
from irl_data import StateEncoder,NUMERIC
from irl_core import LinearReward
from Plot_IRL_Weakness import build_spatial_grid
from Train_IRL import trigger_plots

class SpatialContract(unittest.TestCase):
    def test_reward_gradient_excludes_dynamic_inputs(self):
        encoder=StateEncoder();model=LinearReward(len(encoder.names))
        mask=torch.tensor(reward_mask(encoder.names))
        x=torch.ones((2,len(mask)))
        model(x*mask,torch.tensor([0,1]))[:,1].sum().backward()
        self.assertTrue(torch.all(model.shared.grad[mask==0]==0))
        self.assertTrue(torch.all(model.shared.grad[mask==1]!=0))
        self.assertEqual(mask[encoder.names.index('scaled_elapsed_s')],0)
        self.assertEqual(mask[encoder.names.index('previous_action_0')],0)

    def test_point_rule_does_not_need_trials_or_mdp(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);encoder=StateEncoder()
            encoder.median=np.zeros(len(NUMERIC));encoder.scale=np.ones(len(NUMERIC))
            (p/'config.json').write_text(json.dumps(dict(status='complete',reward_scope=SCOPE)))
            (p/'encoder.json').write_text(json.dumps(dict(encoder=encoder.as_dict())))
            model=LinearReward(len(encoder.names))
            with torch.no_grad():model.shared[encoder.names.index('scaled_concavity')]=2
            torch.save(model.state_dict(),p/'reward.pt')
            rule=SpatialIRLRule(p)
            frame=pd.DataFrame([{**{n:.25 for n in SPATIAL},'floor':'Hall','stage':'to_gate','smoke':'LowSmoke'}])
            self.assertAlmostEqual(float(rule.query_features(frame,'Full')[0]),.5)
            changed=frame.assign(speed_prev=1000,previous_action=1,elapsed_s=99999,trial_id='unseen')
            np.testing.assert_array_equal(rule.query_features(frame,'Full'),rule.query_features(changed,'Full'))
            frame['concavity']=.5
            self.assertAlmostEqual(float(rule.query_features(frame,'Full')[0]),1.)
            self.assertFalse((p/'mdp.npz').exists())

    def test_old_dynamic_models_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'config.json').write_text(json.dumps(dict(status='complete')))
            with self.assertRaisesRegex(ValueError,'retrain older'):
                SpatialIRLRule(p)

    def test_grid_is_full_geometry_without_any_trajectory(self):
        class Spatial:
            domains={'Hall':box(0,0,1.2,1.2)}
            exclusions={'Hall':Polygon()}
            def query(self,x,y,z,radius,**kwargs):
                return SimpleNamespace(status='ok',concavity=x,distance_variation=1,elongation=1,horizontal_size=radius)
        class Decision:
            def query(self,x,y,z,**kwargs):
                return SimpleNamespace(status='ok',stage='to_gate',next_decision_distance=z)
        field=build_spatial_grid(Spatial(),Decision())
        self.assertEqual(len(field),18)
        self.assertEqual(len(field[['x','z']].drop_duplicates()),9)
        self.assertEqual(set(field.smoke),{'HighSmoke','LowSmoke'})
        self.assertFalse({'trial_id','participant_id','elapsed_s'}&set(field.columns))

    def test_auto_trigger_remains_same_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);config=dict(status='complete',reward_scope=SCOPE)
            with patch('Train_IRL.subprocess.run') as launch:
                trigger_plots(run,config)
            args=launch.call_args.args[0]
            self.assertEqual(args[0],sys.executable)
            self.assertEqual(args[args.index('--run')+1],str(run))
            self.assertNotIn('--train',args)
            self.assertEqual(config['plot_status'],'complete')

if __name__=='__main__':unittest.main(verbosity=2)
