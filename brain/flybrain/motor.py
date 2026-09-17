"""Descending-neuron activity -> Minecraft actions (the fly's own reflexes).

Readout neurons (MaleCNS names):
- DNa01, DNa02 (per side): steering; the more active side is the turn direction.
- DNp09 (P9): forward walking.  MDN: backward walking ("moonwalker").
- DNp01 (giant fiber): escape jump.  DNp07 / DNp10: landing / takeoff.
- DNg12_* (aDN1-like): antennal grooming -> stop.
- MN9: proboscis extension -> eat / bite (attack/use).

The decoder returns *logits* per action head, so the learned readout (policy.py) can add
to them; with a zero readout the body is controlled by these reflexes alone.
"""
import numpy as np
import torch

# Action heads and their discrete choices. Must match bot/bot.js.
TURN_DEG = [40.0, 15.0, 5.0, 0.0, -5.0, -15.0, -40.0]    # + = turn left
PITCH_DEG = [20.0, 7.0, 0.0, -7.0, -20.0]                # + = look up
# Crafting macros (executed by bot.js); infeasible options are masked out every cycle.
CRAFTS = ["none", "planks", "sticks", "crafting_table", "wooden_pickaxe", "stone_pickaxe", "furnace", "smelt_iron"]
HEADS = {"move": 3, "turn": len(TURN_DEG), "pitch": len(PITCH_DEG), "jump": 2, "sprint": 2, "attack": 2, "use": 2,
         "craft": len(CRAFTS)}
MASKED = -1e4
MOVE_STOP, MOVE_FORWARD, MOVE_BACK = 0, 1, 2


class MotorMap:
    def __init__(self, conn):
        c = conn
        g = {
            "DNa01_L": c.select(types=["DNa01"], side="L"),
            "DNa01_R": c.select(types=["DNa01"], side="R"),
            "DNa02_L": c.select(types=["DNa02"], side="L"),
            "DNa02_R": c.select(types=["DNa02"], side="R"),
            "DNp09": c.select(types=["DNp09"]),
            "MDN": c.select(types=["MDN"]),
            "DNp01": c.select(types=["DNp01"]),
            "landing": c.select(types=["DNp07", "DNp10"]),
            "groom": c.select(pattern=r"DNg12_.*"),
            "MN9": c.select(types=["MN9"]),
        }
        for name, idx in g.items():
            if len(idx) == 0:
                raise RuntimeError(f"motor group {name} is empty; check the connectome annotations")
        self.groups = g
        # Features for the learned readout: every descending and motor neuron.
        self.feature_idx = c.select(superclass=["descending_neuron", "cb_motor", "vnc_motor"])
        self.names = list(g)
        self.explore = None  # per-body slow turning drive (degrees), see reflex_logits
        self._idx = [torch.as_tensor(g[k], dtype=torch.long) for k in self.names]

    def group_rates(self, counts, ms):
        """counts: [B, N] spikes in a window of `ms`. Returns [B, G] mean rates (Hz)."""
        cols = [counts[:, idx.to(counts.device)].mean(1) for idx in self._idx]
        return torch.stack(cols, 1) * (1000.0 / ms)

    def features(self, counts, ms):
        idx = torch.as_tensor(self.feature_idx, dtype=torch.long, device=counts.device)
        return torch.log1p(counts[:, idx] * (1000.0 / ms))

    def reflex_logits(self, rates, craft_mask=None, alive=None):
        """rates: [B, G] from group_rates -> dict of [B, choices] logits.
        craft_mask: optional [B, len(CRAFTS)] bool tensor of currently feasible crafts."""
        R = {n: rates[:, i] for i, n in enumerate(self.names)}
        lg = torch.log1p
        B = rates.shape[0]
        dev = rates.device

        move = torch.stack([
            0.5 * lg(R["groom"]) - 0.5,              # stop
            1.2 * lg(R["DNp09"]) + 1.8,              # forward (tonic walking drive)
            1.5 * lg(R["MDN"]) - 0.5,                # backward
        ], 1)

        steer = (R["DNa01_L"] + R["DNa02_L"]) - (R["DNa01_R"] + R["DNa02_R"])  # Hz, + = left
        # Escape: turn away from whichever side's looming drove the giant fiber (handled by DNa asym too).
        # Exploration drive: flies walk in persistent bouts with occasional saccadic turns. The
        # steering DNs are often silent, so without this the turn prior is zero-centred noise and
        # sampled turns add up to circles. Ornstein-Uhlenbeck noise, correlation time ~2 s at 10 Hz.
        if self.explore is None or self.explore.shape[0] != B:
            self.explore = torch.zeros(B, device=dev)
        self.explore = 0.95 * self.explore + 4.0 * torch.randn(B, device=dev)
        saccade = (torch.rand(B, device=dev) < 0.02) * torch.sign(torch.randn(B, device=dev)) * 40.0
        self.explore = torch.clamp(self.explore + saccade, -40, 40)
        if alive is not None:
            self.explore = self.explore * alive
        want = torch.clamp(steer * 0.6 + 0.35 * self.explore, -45, 45)
        bins = torch.tensor(TURN_DEG, device=dev)
        turn = -((bins[None, :] - want[:, None]) ** 2) / (2 * 6.0 ** 2)

        pitch = torch.tensor([-1.0, -0.3, 0.5, -0.3, -1.0], device=dev).expand(B, -1)

        escape = lg(R["DNp01"])
        jump = torch.stack([torch.zeros(B, device=dev), 2.0 * escape + 1.0 * lg(R["landing"]) - 3.0], 1)
        sprint = torch.stack([torch.zeros(B, device=dev), 2.0 * escape - 2.0], 1)
        feed = lg(R["MN9"])
        attack = torch.stack([torch.zeros(B, device=dev), 1.5 * feed - 3.0], 1)
        use = torch.stack([torch.zeros(B, device=dev), 1.5 * feed - 3.5], 1)
        # A fly has no crafting reflex: "none" is strongly preferred until the readout learns otherwise.
        craft = torch.zeros(B, len(CRAFTS), device=dev)
        craft[:, 0] = 3.0
        if craft_mask is not None:
            craft = craft + (~craft_mask).float() * MASKED
        return {"move": move, "turn": turn, "pitch": pitch, "jump": jump, "sprint": sprint, "attack": attack, "use": use,
                "craft": craft}
