"""Tech-tree speedrun task: milestones, split times, rewards and the task-state vector.

Milestones must be reached in order-independent fashion but are scored by how fast
each one is reached after the episode starts (a speedrun split).
"""
import math

import torch

from .motor import CRAFTS

KEY_ITEMS = ["log", "planks", "stick", "crafting_table", "wooden_pickaxe", "cobblestone", "stone_pickaxe",
             "raw_iron", "furnace", "iron_ingot", "coal"]

# (milestone, reward, par time in seconds): beating par earns up to 2x the reward.
MILESTONES = [
    ("log", 2.0, 60),
    ("logs3", 3.0, 150),
    ("planks", 2.0, 170),
    ("crafting_table", 5.0, 200),
    ("wooden_pickaxe", 10.0, 260),
    ("cobblestone", 10.0, 330),
    ("stone_pickaxe", 15.0, 400),
    ("raw_iron", 25.0, 900),
    ("furnace", 15.0, 950),
    ("iron_ingot", 40.0, 1100),
]
MILESTONE_NAMES = [m[0] for m in MILESTONES]
TARGET_KINDS = ["none", "log", "leaves", "stone", "iron_ore", "ground", "entity", "other"]
N_TASK = len(KEY_ITEMS) + len(MILESTONES) + 3 + len(TARGET_KINDS) + 4


def target_kind(target):
    if not target:
        return "none"
    if target.get("kind") == "entity":
        return "entity"
    n = target.get("name", "")
    if n.endswith("_log") or n.endswith("_wood"):
        return "log"
    if n.endswith("_leaves"):
        return "leaves"
    if "iron_ore" in n:
        return "iron_ore"
    if n in ("stone", "cobblestone", "deepslate", "andesite", "diorite", "granite", "tuff") or n.endswith("coal_ore"):
        return "stone"
    if n in ("dirt", "grass_block", "sand", "gravel", "coarse_dirt", "podzol", "snow", "snow_block"):
        return "ground"
    return "other"


def task_vector(obs, episode_limit_s):
    v = []
    items = obs.get("items", {})
    v += [math.log1p(items.get(k, 0)) / 3.0 for k in KEY_ITEMS]
    progress = obs.get("progress", {})
    v += [1.0 if m in progress else 0.0 for m in MILESTONE_NAMES]
    near = obs.get("near", {})
    v += [float(near.get("crafting_table", 0)), float(near.get("furnace", 0)), float(obs.get("busy", 0))]
    kind = target_kind(obs.get("target"))
    v += [1.0 if kind == k else 0.0 for k in TARGET_KINDS]
    v += [obs.get("health", 20) / 20, obs.get("food", 20) / 20, obs.get("day", 1.0),
          min(1.0, obs.get("episode_s", 0) / episode_limit_s)]
    return v


def batch_task(obs_list, device, episode_limit_s=900.0):
    """Returns (task [B, N_TASK], craft_mask [B, len(CRAFTS)] bool)."""
    B = len(obs_list)
    task = torch.zeros(B, N_TASK)
    mask = torch.zeros(B, len(CRAFTS), dtype=torch.bool)
    mask[:, 0] = True
    for b, o in enumerate(obs_list):
        if o is None:
            continue
        task[b] = torch.tensor(task_vector(o, episode_limit_s))
        ok = o.get("craftable")
        if ok and not o.get("busy"):
            mask[b, 1:] = torch.tensor([bool(x) for x in ok[1:]])
    return task.to(device), mask.to(device)


class SpeedrunRewarder:
    """Milestone splits + dopamine-like dense reward (dopamine.py) for each body."""

    def __init__(self, n, hz=10.0):
        from .dopamine import Dopamine
        self.n = n
        self.dopamine = [Dopamine(hz) for _ in range(n)]
        self.reset_all()

    def reset_all(self):
        self.done_ms = [set() for _ in range(self.n)]
        self.awaiting = [0] * self.n  # >0: cycles spent waiting for the body to confirm a fresh attempt
        for b in range(self.n):
            self.reset(b)

    def reset(self, b):
        self.done_ms[b] = set()
        self.awaiting[b] = 1
        self.dopamine[b].reset()

    def needs_resend(self, b):
        """The reset message may have been lost (e.g. the body reconnected)."""
        return self.awaiting[b] > 100

    def __call__(self, b, obs):
        """Returns (reward, died, new_milestones [(name, split_s)], dopamine components)."""
        if obs is None:
            return 0.0, False, [], {}
        if self.awaiting[b]:
            # observations sent before the bot processed the reset still carry the old splits
            if obs.get("episode_s", 1e9) > 3.0 or obs.get("progress"):
                self.awaiting[b] += 1
                return 0.0, False, [], {}
            self.awaiting[b] = 0
        milestone = 0.0
        new = []
        for name, reward, par in MILESTONES:
            split = obs.get("progress", {}).get(name)
            if split is not None and name not in self.done_ms[b]:
                self.done_ms[b].add(name)
                milestone += reward * (1.0 + max(0.0, 1.0 - split / par))
                new.append((name, split))
        r, died, comps = self.dopamine[b].step(obs, milestone)
        return r, died, new, comps

    def complete(self, b):
        return len(self.done_ms[b]) == len(MILESTONES)

    def drive(self):
        """[(pam, ppl1)] per body, 0..1, for the fly's dopamine neurons."""
        return [(d.pam, d.ppl1) for d in self.dopamine]
