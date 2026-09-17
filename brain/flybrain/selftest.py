"""Offline check of the whole brain pipeline with synthetic observations (no Minecraft needed).

    python -m flybrain.selftest
"""
import math
import time

import numpy as np

from .bridge import Agent
from .senses import EYE_COLS, EYE_ROWS
from .train import PPO


def fake_obs(rng, loom_left=0.0, sugar=0.0, touch=0.0):
    rays = EYE_ROWS * EYE_COLS
    return {
        "dead": False, "eye": {s: rng.random(rays).tolist() for s in "LR"},
        "eye_delta": {s: (rng.random(rays) * 0.05).tolist() for s in "LR"},
        "loom": {"L": loom_left, "R": 0.0}, "food_odor": {"L": 0.2, "R": 0.0}, "danger_odor": {"L": 0, "R": 0},
        "plant_odor": {"L": 0.1, "R": 0.1}, "taste_sugar": sugar, "taste_bitter": 0.0,
        "touch": {"L": touch, "R": touch}, "wind": 0.0, "hurt": 0.0, "noise": 0.0, "health": 20, "food": 10,
        "pos": [0, 64, 0], "yaw": 0, "pitch": 0, "on_ground": True, "in_water": False, "day": 1.0,
        "stats": {"mined": 0, "collected": 0, "eaten": 0, "damage": 0, "deaths": 0, "logs": 0},
        "inventory": {"logs": 0, "food": 0, "total": 0}, "target": None,
    }


def main():
    rng = np.random.default_rng(0)
    agent = Agent(4, window_ms=100)
    scenarios = {"baseline": {}, "loom left": {"loom_left": 1.0}, "sugar": {"sugar": 1.0}, "touch": {"touch": 1.0}}
    obs = [fake_obs(rng, **kw) for kw in scenarios.values()]
    agent.think(obs)  # warm-up / CUDA graph capture
    t0 = time.perf_counter()
    for _ in range(10):
        out = agent.think(obs)
    dt = (time.perf_counter() - t0) / 10
    print(f"think(): {dt * 1000:.0f} ms per cycle for 4 bodies x 100 ms neural time ({0.1 / dt:.2f}x real time)")
    names = agent.motor.names
    print(f"{'scenario':12s} " + " ".join(f"{n[:7]:>7s}" for n in names))
    for i, s in enumerate(scenarios):
        print(f"{s:12s} " + " ".join(f"{out['group_rates'][i, j]:7.0f}" for j in range(len(names))))
    print("reflex agreement (untrained readout, should be ~sampling noise only):", out["reflex_agree"].tolist())

    # Dopamine: reward drives PAM, punishment drives PPL1; check the mushroom-body output responds.
    agent.dopamine = [(1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)]
    out = agent.think([fake_obs(rng) for _ in range(4)])
    agent.dopamine = None
    counts = out["counts"].cpu().numpy()
    mbon = agent.conn.select(cls="MBON")
    for i, label in enumerate(["reward", "punishment", "none"]):
        rate = lambda idx: counts[i, idx].mean() * 10  # 100 ms window -> Hz
        print(f"dopamine {label:10s}: PAM {rate(agent.senses.groups['dan_reward']):6.1f} Hz, "
              f"PPL1 {rate(agent.senses.groups['dan_punish']):6.1f} Hz, MBONs {rate(mbon):5.1f} Hz")
    assert counts[0, agent.senses.groups["dan_reward"]].mean() > counts[2, agent.senses.groups["dan_reward"]].mean()
    assert counts[1, agent.senses.groups["dan_punish"]].mean() > counts[2, agent.senses.groups["dan_punish"]].mean()

    from .dopamine import Dopamine
    circ, walker = Dopamine(), Dopamine()
    for t in range(80):
        o1 = fake_obs(rng); o1["pos"] = [math.cos(t / 3), 64, math.sin(t / 3)]; o1["yaw"] = t / 3
        o2 = fake_obs(rng); o2["pos"] = [t * 0.4, 64, 0]; o2["nearest_log_dist"] = 40 - t * 0.4
        _, _, c1 = circ.step(o1)
        _, _, c2 = walker.step(o2)
    print(f"circling fly: circling={c1['circling']:.3f} novelty={c1['novelty']:.3f} | "
          f"straight walker towards a tree: circling={c2['circling']:.3f} approach={c2['approach']:.3f}")
    assert c1["circling"] < 0 and c2["circling"] == 0 and c2["approach"] > 0

    ppo = PPO(agent, horizon=8, minibatch=16)
    for _ in range(9):
        out = agent.think(obs)
        ppo.add(out, rewards=rng.random(4).tolist(), dones=[0.0] * 4, mask=[1.0] * 4)
    stats = ppo.update()
    print("PPO update ok:", {k: round(v, 4) for k, v in stats.items()})
    assert all(np.isfinite(v) for v in stats.values())
    print("selftest passed")


if __name__ == "__main__":
    main()
