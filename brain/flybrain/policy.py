"""Learned readout on top of the frozen connectome.

logits(head) = reflex_logits(head) + readout(head)

The readout sees only the firing rates of descending and motor neurons, the same
bottleneck through which a real fly brain talks to its body. Its output heads start
at zero, so an untrained readout is exactly the fly's reflexes.
"""
import torch
from torch import nn
from torch.distributions import Categorical

from .motor import CRAFTS, HEADS, PITCH_DEG, TURN_DEG


class Readout(nn.Module):
    """features: descending/motor neuron rates. task: small game-state vector (inventory,
    milestones, looked-at block) that the fly brain itself has no neurons for."""

    def __init__(self, n_features, n_task=0, hidden=96):
        super().__init__()
        self.norm = nn.LayerNorm(n_features)
        self.proj = nn.Linear(n_features, hidden)
        self.task = nn.Linear(n_task, hidden, bias=False) if n_task else None
        self.heads = nn.ModuleDict({k: nn.Linear(hidden, n) for k, n in HEADS.items()})
        self.value = nn.Linear(hidden, 1)
        for h in self.heads.values():
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    def forward(self, features, reflex, task=None):
        h = self.proj(self.norm(features))
        if self.task is not None and task is not None:
            h = h + self.task(task)
        z = torch.tanh(h)
        logits = {k: reflex[k] + self.heads[k](z) for k in HEADS}
        return logits, self.value(z).squeeze(-1)


def sample(logits, greedy=False):
    """Returns (actions {head: [B]}, log_prob [B], entropy [B])."""
    acts, logp, ent = {}, 0, 0
    for k in HEADS:
        d = Categorical(logits=logits[k])
        a = logits[k].argmax(-1) if greedy else d.sample()
        acts[k] = a
        logp = logp + d.log_prob(a)
        ent = ent + d.entropy()
    return acts, logp, ent


def evaluate(logits, acts):
    logp, ent = 0, 0
    for k in HEADS:
        d = Categorical(logits=logits[k])
        logp = logp + d.log_prob(acts[k])
        ent = ent + d.entropy()
    return logp, ent


def to_commands(acts):
    """Batched action indices -> list of JSON-able dicts for bot.js."""
    cpu = {k: v.tolist() for k, v in acts.items()}
    return [
        {
            "move": cpu["move"][b],
            "turn": TURN_DEG[cpu["turn"][b]],
            "pitch": PITCH_DEG[cpu["pitch"][b]],
            "jump": cpu["jump"][b],
            "sprint": cpu["sprint"][b],
            "attack": cpu["attack"][b],
            "use": cpu["use"][b],
            "craft": CRAFTS[cpu["craft"][b]],
        }
        for b in range(len(cpu["move"]))
    ]
