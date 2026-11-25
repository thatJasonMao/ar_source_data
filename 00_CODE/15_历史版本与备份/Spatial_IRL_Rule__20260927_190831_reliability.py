"""Spatial-only reward contract and trajectory-independent point queries."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import torch
from irl_data import StateEncoder, NUMERIC, GUIDANCE, STAGES
from irl_core import LinearReward

SPATIAL = NUMERIC[:5]
SCOPE = 'spatial_reward_v1'


def reward_mask(names):
    allowed = {'bias','high_smoke',*('scaled_'+n for n in SPATIAL)}
    return np.asarray([name in allowed or name.startswith(('floor_','stage_')) for name in names],dtype=np.float32)


class SpatialIRLRule:
    """Needs only the fitted rule/encoder, never trials, trajectories or an MDP."""
    def __init__(self, run):
        run=Path(run)
        config=json.loads((run/'config.json').read_text(encoding='utf-8'))
        if config.get('status')!='complete' or config.get('reward_scope')!=SCOPE:
            raise ValueError('Requires a completed spatial_reward_v1 model; retrain older dynamic-reward models')
        saved=json.loads((run/'encoder.json').read_text(encoding='utf-8'))['encoder']
        self.encoder=StateEncoder()
        self.encoder.median=np.asarray(saved['median']);self.encoder.scale=np.asarray(saved['scale'])
        if self.encoder.names!=saved['names']:raise ValueError('Encoder schema mismatch')
        model=LinearReward(len(self.encoder.names))
        model.load_state_dict(torch.load(run/'reward.pt',map_location='cpu',weights_only=True))
        delta=model.deviation.detach().numpy()
        self.weights=model.shared.detach().numpy()[None,:]+delta-delta.mean(0,keepdims=True)
        self.mask=reward_mask(self.encoder.names)
        if np.any(np.abs(self.weights[:,self.mask==0])>1e-8):
            raise ValueError('Reward still depends on nonspatial inputs')

    def query_features(self, frame, guidance):
        if guidance not in GUIDANCE:raise ValueError('Unsupported guidance')
        required=SPATIAL+['floor','stage','smoke']
        if not set(required)<=set(frame):raise ValueError('Missing spatial inputs')
        if not np.isfinite(frame[SPATIAL].to_numpy(float)).all():raise ValueError('Invalid spatial inputs')
        for key,allowed in [('floor',['Hall','Platform']),('stage',STAGES),('smoke',['HighSmoke','LowSmoke'])]:
            if not frame[key].isin(allowed).all():raise ValueError('Unknown '+key)
        # Encoder placeholders are multiplied by an exact zero reward mask.
        # They are not assumptions about a pedestrian or a fabricated history.
        state=frame.copy()
        for name in NUMERIC[5:]:state[name]=0.
        state['previous_action']=-1
        features=self.encoder.transform(state)*self.mask
        return features@self.weights[GUIDANCE.index(guidance)]

    def query_position(self,x,y,z,*,guidance,smoke,floor=None):
        if smoke not in {'HighSmoke','LowSmoke'}:raise ValueError('Unknown smoke')
        source=Path(__file__).resolve().parent.parent/'0_RL_SOURCE'
        if str(source) not in sys.path:sys.path.insert(0,str(source))
        from spatial_isovist_core import get_engine
        from Analyze_NextDecisionDistance import get_engine as get_decision
        s=get_engine().query(x,y,z,7.5 if smoke=='HighSmoke' else 12.5,floor=floor)
        d=get_decision().query(x,y,z,floor=floor)
        if s.status!='ok' or d.status!='ok':return dict(status=s.status if s.status!='ok' else d.status)
        if d.stage=='arrived_exit':return dict(status='task_complete')
        row={name:getattr(s,name) for name in SPATIAL[:4]}
        row.update(next_decision_distance=d.next_decision_distance,floor=d.floor,stage=d.stage,smoke=smoke)
        score=float(self.query_features(pd.DataFrame([row]),guidance)[0])
        return dict(status='ok',guidance=guidance,smoke=smoke,relative_information_utility=score,
                    interpretation='candidate guidance weakness index; not a failure probability',**row)
