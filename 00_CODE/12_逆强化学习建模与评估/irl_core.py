"""Finite-horizon soft-policy likelihood IRL; no separate policy optimizer.

The reward is a restricted, gauge-fixed *relative action utility*, not a
uniquely identified psychological potential. Transitions are fixed estimates.
"""
import math
import torch


class LinearReward(torch.nn.Module):
    def __init__(self, n_features, n_guidance=4):
        super().__init__()
        self.shared = torch.nn.Parameter(torch.zeros(n_features))
        self.deviation = torch.nn.Parameter(torch.zeros(n_guidance,n_features))

    def forward(self, features, guidance):
        deviation = self.deviation-self.deviation.mean(0,keepdim=True)
        contrast = (features*(self.shared+deviation[guidance])).sum(1)
        # Fix temperature=1 and mean immediate reward across actions=0.
        return torch.stack([-contrast/2,contrast/2],dim=1)


def soft_policy(reward, transitions, horizon=25, gamma=.98, terminal_index=None):
    """Exactly horizon backup steps; deterministic differentiable computation.

    V_0=0; Q_h=r+gamma P V_(h-1); V_h=logmeanexp_a Q_h.
    A uniform action prior (KL regularization) avoids a free entropy bonus for
    delaying task completion. Absorbing task completion has V=0.
    Returns a horizon-H first-step log policy, not an independently optimized agent.
    """
    if horizon < 1 or not 0 <= gamma <= 1:
        raise ValueError('Invalid planning horizon/discount')
    value = torch.zeros(reward.shape[0],dtype=reward.dtype,device=reward.device)
    for _ in range(horizon):
        q = reward+gamma*torch.einsum('sak,k->sa',transitions,value)
        value = torch.logsumexp(q,dim=1)-math.log(reward.shape[1])
        if terminal_index is not None:
            keep = torch.arange(len(value),device=value.device) != terminal_index
            value = torch.where(keep,value,torch.zeros_like(value))
    return torch.log_softmax(q,dim=1),q


def weighted_nll(log_policy, counts):
    return -(counts*log_policy).sum()/counts.sum().clamp_min(1e-12)


def expected_burden(policy, transitions, event_cost, horizon, step_s):
    """Finite-window expected joint-event seconds, under fitted human policy."""
    cost = (policy*event_cost).sum(1)
    kernel = torch.einsum('sa,sak->sk',policy,transitions)
    value = torch.zeros_like(cost)
    for _ in range(horizon):
        value = cost*step_s+kernel@value
    return cost,value
