"""Dopamine-like reward: dense intrinsic reward / punishment plus a reward-prediction error.

Real flies have two main dopamine systems feeding the mushroom body: PAM neurons signal
reward, PPL1 neurons signal punishment. Here each body gets a `Dopamine` tracker that:

  * turns observations into reward (r+) and punishment (r-) components each cycle
      novelty    - count-based curiosity bonus for new 2-block cells (circling pays nothing)
      approach   - getting closer to the next useful resource (logs, later stone)
      in_view    - crosshair on that resource
      progress   - blocks mined, items picked up
      circling   - little net displacement or lots of spinning over the last 5 s
      damage     - health lost
  * keeps a running baseline so it can report a reward-prediction error (RPE)
  * exposes a decaying PAM / PPL1 drive used to fire the fly's real dopamine neurons

Milestone rewards (task.py) are added on top by the trainer and also count as dopamine.
"""
import math
from collections import deque

from .task import target_kind

DOPAMINE_WEIGHTS = {
    "novelty": 0.04,       # per sqrt-count, new 2-block cell
    "approach": 0.03,      # per block closer to the next resource
    "in_view": 0.004,      # per cycle with the resource under the crosshair
    "mined": 0.15,
    "collected": 0.25,
    "circling": -0.01,     # per cycle while stuck / spinning
    "damage": -0.08,       # per half-heart
    "death": -3.0,
    "step": -0.0005,       # time pressure
}
WINDOW_S = 5.0
CELL = 2.0


class Dopamine:
    def __init__(self, hz=10.0):
        self.hz = hz
        self.reset()

    def reset(self):
        self.visits = {}
        self.trail = deque(maxlen=int(WINDOW_S * self.hz))
        self.prev_dist = None
        self.prev_stats = None
        self.prev_dead = False
        self.baseline = 0.0
        self.pam = 0.0     # decaying reward drive, 0..1
        self.ppl1 = 0.0    # decaying punishment drive, 0..1
        self.last = {}

    def _resource(self, obs):
        items = obs.get("items", {})
        has_pick = items.get("wooden_pickaxe", 0) or items.get("stone_pickaxe", 0)
        if has_pick and "cobblestone" not in obs.get("progress", {}):
            return "stone", obs.get("nearest_stone_dist")
        return "log", obs.get("nearest_log_dist")

    def step(self, obs, milestone_reward=0.0):
        """Returns (reward, died, components). Call once per control cycle per body."""
        W = DOPAMINE_WEIGHTS
        c = {k: 0.0 for k in W}
        if obs is None:
            return 0.0, False, c
        dead = bool(obs.get("dead"))
        died = dead and not self.prev_dead
        self.prev_dead = dead
        if died:
            c["death"] = W["death"]
        if dead:
            return self._finish(c, milestone_reward, died)

        c["step"] = W["step"]
        x, y, z = obs["pos"]
        cell = (math.floor(x / CELL), math.floor(z / CELL))
        n = self.visits.get(cell, 0) + 1
        self.visits[cell] = n
        c["novelty"] = W["novelty"] / math.sqrt(n) if n <= 3 else 0.0

        kind, dist = self._resource(obs)
        if dist is not None and self.prev_dist is not None and self.prev_dist[0] == kind:
            c["approach"] = W["approach"] * max(-1.0, min(1.0, self.prev_dist[1] - dist))
        self.prev_dist = (kind, dist) if dist is not None else None
        target = target_kind(obs.get("target"))
        if target == kind:
            c["in_view"] = W["in_view"]

        s, p = obs.get("stats", {}), self.prev_stats
        self.prev_stats = s
        if p:
            d = {k: max(0, s.get(k, 0) - p.get(k, 0)) for k in ("mined", "collected", "damage")}
            c["mined"] = W["mined"] * d["mined"]
            c["collected"] = W["collected"] * d["collected"]
            c["damage"] = W["damage"] * d["damage"]

        self.trail.append((x, z, obs.get("yaw", 0.0)))
        if len(self.trail) == self.trail.maxlen and not obs.get("busy"):
            x0, z0, _ = self.trail[0]
            disp = math.hypot(x - x0, z - z0)
            spin = 0.0
            yaws = [t[2] for t in self.trail]
            for a, b in zip(yaws, yaws[1:]):
                spin += abs((b - a + math.pi) % (2 * math.pi) - math.pi)
            if disp < 1.5 or spin > math.radians(540):
                c["circling"] = W["circling"]
        return self._finish(c, milestone_reward, died)

    def _finish(self, c, milestone_reward, died):
        r = sum(c.values()) + milestone_reward
        c["milestone"] = milestone_reward
        rpe = r - self.baseline
        self.baseline += 0.02 * (r - self.baseline)
        # Burst size ~ surprise; decays over a few cycles so the spikes are visible.
        self.pam = max(self.pam * 0.6, min(1.0, max(0.0, rpe) * 8.0))
        self.ppl1 = max(self.ppl1 * 0.6, min(1.0, max(0.0, -rpe) * 8.0))
        c["rpe"] = rpe
        self.last = c
        return float(r), died, c
