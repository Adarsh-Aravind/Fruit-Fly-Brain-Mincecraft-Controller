"""Stimulus -> response experiments on the simulated connectome.

    python -m flybrain.probe screen-taste     # which gustatory types drive feeding (MN9) vs bitter output
    python -m flybrain.probe check            # reflex circuits used by the Minecraft body
"""
import argparse

import numpy as np
import torch

from .connectome import Connectome
from .lif import Brain


def respond(conn, brain, stim_groups, readouts, ms=500, rate=150.0):
    """stim_groups: list (len <= B) of index arrays. Returns {readout: [rate Hz per stim]}."""
    B = brain.b
    brain.reset()
    r = torch.zeros(B, conn.n, device=brain.device)
    for i, idx in enumerate(stim_groups):
        r[i, torch.as_tensor(idx, dtype=torch.long)] = rate
    brain.set_input(r)
    counts = brain.run(ms).cpu().numpy()
    out = {}
    for name, idx in readouts.items():
        out[name] = counts[: len(stim_groups), idx].mean(1) / ms * 1000 if len(idx) else np.zeros(len(stim_groups))
    return out


def screen_taste(conn, brain):
    grn = conn.select(cls="gustatory")
    types = sorted({t for t in conn.type[grn] if t})
    readouts = {
        "MN9": conn.select(types=["MN9"]),
        "sugarSEL": np.flatnonzero([("Sugar SEL" in s) for s in conn.synonyms]),
        "bitterSEL": np.flatnonzero([("Bitter-SEL" in s) for s in conn.synonyms]),
    }
    rows = []
    for i in range(0, len(types), brain.b):
        chunk = types[i : i + brain.b]
        res = respond(conn, brain, [conn.select(types=[t]) for t in chunk], readouts)
        for j, t in enumerate(chunk):
            rows.append((t, len(conn.select(types=[t])), *(res[k][j] for k in readouts)))
    rows.sort(key=lambda r: -r[2])
    print(f"{'type':14s} {'n':>4s} " + " ".join(f"{k:>9s}" for k in readouts))
    for r in rows:
        print(f"{r[0]:14s} {r[1]:4d} " + " ".join(f"{x:9.1f}" for x in r[2:]))


SUGAR_PROBE = ["LB3c", "LB3d"]


def screen_bitter(conn, brain):
    """Bitter taste suppresses sugar-evoked feeding: find gustatory types that silence MN9."""
    sugar = conn.select(types=SUGAR_PROBE)
    grn = conn.select(cls="gustatory")
    types = ["(sugar only)"] + sorted({t for t in conn.type[grn] if t and t not in SUGAR_PROBE})
    mn9 = {"MN9": conn.select(types=["MN9"])}
    rows = []
    for i in range(0, len(types), brain.b):
        chunk = types[i : i + brain.b]
        stims = [sugar if t == "(sugar only)" else np.concatenate([sugar, conn.select(types=[t])]) for t in chunk]
        res = respond(conn, brain, stims, mn9)
        rows += list(zip(chunk, res["MN9"]))
    rows.sort(key=lambda r: r[1])
    for t, r in rows:
        print(f"{t:14s} MN9 {r:6.1f} Hz")


def check(conn, brain):
    from .senses import SensoryMap
    from .motor import MotorMap
    sm, mm = SensoryMap(conn), MotorMap(conn)
    readouts = {name: idx for name, idx in mm.groups.items()}
    tests = [(name, sm.groups[name]) for name in sm.groups]
    tests.insert(0, ("(none)", np.array([], dtype=np.int64)))
    names = list(readouts)
    print(f"{'stimulus':22s} " + " ".join(f"{n[:8]:>8s}" for n in names))
    for i in range(0, len(tests), brain.b):
        chunk = tests[i : i + brain.b]
        res = respond(conn, brain, [idx for _, idx in chunk], readouts)
        for j, (sname, _) in enumerate(chunk):
            print(f"{sname:22s} " + " ".join(f"{res[n][j]:8.1f}" for n in names))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["screen-taste", "screen-bitter", "check"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--shuffled", action="store_true", help="use a degree-preserving shuffled connectome")
    args = ap.parse_args()
    conn = Connectome.load()
    if args.shuffled:
        conn = conn.shuffled()
    brain = Brain(conn, batch=args.batch)
    {"screen-taste": screen_taste, "screen-bitter": screen_bitter, "check": check}[args.what](conn, brain)


if __name__ == "__main__":
    main()
