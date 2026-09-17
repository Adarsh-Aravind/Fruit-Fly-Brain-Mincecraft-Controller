"""Minecraft observations -> Poisson firing rates of real MaleCNS sensory neurons.

Group choices:
- Photoreceptors R1-R6 per eye get a luminance image (each neuron samples one ray).
- LC4 / LPLC2 looming detectors per side get a looming signal (approaching mobs).
- Olfactory receptor neurons per antenna: food odour -> DM1/DM4/VA2/DM2 (fruity esters),
  danger odour -> V (CO2) / DA2 (geosmin), both innately aversive for flies.
- Sugar and bitter taste types were identified by simulation (`python -m flybrain.probe
  screen-taste` / `screen-bitter`): sugar types drive the proboscis motor neuron MN9,
  bitter types suppress sugar-evoked MN9 activity.
- Bristle mechanosensory neurons by side for collisions, Johnston's organ C/E
  (gravity/wind) for falling and knockback, JO-A/B (vibration) plus bristles for damage.
"""
import numpy as np
import torch

SUGAR_TYPES = ["LB3c", "LB3d", "LB2a", "LB4a", "LB4b", "claw_tpGRN", "dorsal_tpGRN"]
BITTER_TYPES = ["LB1b", "LB1c", "LB1e", "LgAG3", "LgAG5", "PhG8", "PhG16"]
FOOD_ORN = ["ORN_DM1", "ORN_DM4", "ORN_VA2", "ORN_DM2"]
DANGER_ORN = ["ORN_V", "ORN_DA2"]
PLANT_ORN = ["ORN_DL5", "ORN_VM7d", "ORN_VM7v"]  # green-leaf / plant volatiles: wood and dropped items
EYE_ROWS, EYE_COLS = 12, 16  # rays per eye (180 x 140 deg panorama), must match bot/bot.js

MAX_RATE = 150.0  # Hz, as in Shiu et al.


class SensoryMap:
    def __init__(self, conn):
        c = conn
        g = {}
        for s in "LR":
            g[f"eye_{s}"] = c.select(types=["R1-R6"], side=s)
            g[f"loom_{s}"] = np.concatenate([c.select(types=["LC4"], side=s), c.select(types=["LPLC2"], side=s)])
            g[f"food_odor_{s}"] = c.select(types=FOOD_ORN, side=s)
            g[f"danger_odor_{s}"] = c.select(types=DANGER_ORN, side=s)
            g[f"plant_odor_{s}"] = c.select(types=PLANT_ORN, side=s)
            g[f"touch_{s}"] = c.select(types=["BM_InOm", "BM"], side=s)
        g["sugar"] = c.select(types=SUGAR_TYPES)
        g["bitter"] = c.select(types=BITTER_TYPES)
        # Dopamine neurons of the mushroom body: PAM = reward, PPL1/PPL2 = punishment.
        g["dan_reward"] = c.select(pattern=r"PAM\d+")
        g["dan_punish"] = c.select(pattern=r"PPL[12]\d+")
        g["wind"] = c.select(pattern=r"JO-[CE].*")
        g["vibration"] = c.select(pattern=r"JO-[AB].*")
        self.groups = g
        self.n = c.n
        for name, idx in g.items():
            if len(idx) == 0:
                raise RuntimeError(f"sensory group {name} is empty; check the connectome annotations")
        # Each photoreceptor samples one ray of its eye (fixed pseudo-random retinotopy).
        rng = np.random.default_rng(0)
        rays = EYE_ROWS * EYE_COLS
        self.eye_ray = {s: rng.permutation(np.arange(len(g[f"eye_{s}"])) % rays) for s in "LR"}

    def rates(self, obs_list, device, dopamine=None):
        """obs_list: list of observation dicts (one per body); dopamine: optional [(pam, ppl1)]
        drives in 0..1 per body. Returns [B, N] rates (Hz)."""
        B = len(obs_list)
        r = np.zeros((B, self.n), np.float32)
        g = self.groups
        for b, o in enumerate(obs_list):
            if o is None or o.get("dead"):
                continue
            hunger = 1.0 - o.get("food", 20) / 20.0
            for s in "LR":
                lum = np.asarray(o["eye"][s], np.float32)
                dl = np.asarray(o.get("eye_delta", {}).get(s, np.zeros_like(lum)), np.float32)
                ray = self.eye_ray[s]
                r[b, g[f"eye_{s}"]] = np.clip(5 + 60 * lum[ray] + 400 * np.abs(dl[ray]), 0, MAX_RATE)
                r[b, g[f"loom_{s}"]] = MAX_RATE * min(1.0, o["loom"][s])
                r[b, g[f"food_odor_{s}"]] = MAX_RATE * min(1.0, o["food_odor"][s])
                r[b, g[f"danger_odor_{s}"]] = MAX_RATE * min(1.0, o["danger_odor"][s])
                r[b, g[f"plant_odor_{s}"]] = MAX_RATE * min(1.0, o["plant_odor"][s])
                r[b, g[f"touch_{s}"]] = MAX_RATE * min(1.0, o["touch"][s] + o["hurt"])
            r[b, g["sugar"]] = MAX_RATE * min(1.0, o["taste_sugar"] * (0.4 + 0.6 * hunger))
            r[b, g["bitter"]] = MAX_RATE * min(1.0, o["taste_bitter"])
            r[b, g["wind"]] = MAX_RATE * min(1.0, o["wind"])
            r[b, g["vibration"]] = MAX_RATE * min(1.0, o["hurt"] + 0.3 * o.get("noise", 0))
            pam, ppl1 = dopamine[b] if dopamine is not None and b < len(dopamine) else (0.0, 0.0)
            r[b, g["dan_reward"]] = 2.0 + MAX_RATE * pam
            r[b, g["dan_punish"]] = 2.0 + MAX_RATE * ppl1
        return torch.from_numpy(r).to(device, non_blocking=True)

    def summary(self, rates_row):
        """Mean rate per group for one body (for the dashboard)."""
        return {k: float(rates_row[idx].mean()) for k, idx in self.groups.items()}
