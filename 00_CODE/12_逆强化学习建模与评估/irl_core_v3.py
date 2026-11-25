"""V1 soft-policy IRL with real-duration discounting for full-frame states."""
import math
import torch
from irl_core import LinearReward, weighted_nll

def soft_policy(reward, transitions, horizon=25, gamma=.98, terminal_index=None, mean_dt=None):
    if horizon < 1 or not 0 <= gamma <= 1: raise ValueError('Invalid horizon/discount')
    # gamma denotes the V1 discount per 0.2 seconds, even at native frame rate.
    discount = gamma if mean_dt is None else torch.pow(torch.full_like(mean_dt,gamma),mean_dt/.2)
    value = torch.zeros(reward.shape[0],dtype=reward.dtype,device=reward.device)
    for _ in range(horizon):
        q=reward+discount*torch.einsum('sak,k->sa',transitions,value)
        value=torch.logsumexp(q,dim=1)-math.log(reward.shape[1])
        if terminal_index is not None:
            value=torch.where(torch.arange(len(value),device=value.device)!=terminal_index,value,torch.zeros_like(value))
    return torch.log_softmax(q,dim=1),q
