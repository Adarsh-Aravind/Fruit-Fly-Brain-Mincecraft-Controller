"""PPO on the readout only; the 165k-neuron connectome stays frozen.

    python -m flybrain.train --task speedrun --bodies 4 --run speedrun
    python -m flybrain.train --task survival --bodies 4 --run default

Speedrun task: race through the tech tree (see task.py), scored by split times.
Survival curriculum (driven over RCON):
  explore  - peaceful, permanent daytime: wander, don't get hurt, pick things up
  gather   - peaceful, daytime: logs are worth a lot
  survive  - normal difficulty, day/night cycle: stay alive, eat
"""
import argparse
import asyncio
import json
import time

import numpy as np
import torch

from .bridge import Agent, BodyServer, Dashboard, control_loop
from .motor import HEADS
from .paths import RUNS
from .policy import evaluate
from .rcon import Rcon
from .task import SpeedrunRewarder

STAGES = {
    "explore": dict(commands=["difficulty peaceful", "gamerule doDaylightCycle false", "time set day", "weather clear"],
                    w=dict(alive=0.002, explore=0.05, collected=0.3, logs=1.0, mined=0.05, eaten=0.1, damage=-0.1, death=-3.0)),
    "gather": dict(commands=["difficulty peaceful", "gamerule doDaylightCycle false", "time set day", "weather clear"],
                   w=dict(alive=0.001, explore=0.02, collected=0.3, logs=2.0, mined=0.2, eaten=0.1, damage=-0.1, death=-3.0)),
    "survive": dict(commands=["difficulty normal", "gamerule doDaylightCycle true"],
                    w=dict(alive=0.004, explore=0.01, collected=0.2, logs=1.0, mined=0.1, eaten=0.3, damage=-0.15, death=-5.0)),
}
STAGES["speedrun"] = dict(commands=["difficulty peaceful", "gamerule doDaylightCycle false", "time set day", "weather clear"], w={})
STAGE_ORDER = ["explore", "gather", "survive"]  # survival curriculum; the speedrun task stays in "speedrun"
WORLD_SETUP = ["gamerule doImmediateRespawn true", "gamerule sendCommandFeedback false", "gamerule announceAdvancements false",
               "gamerule spawnRadius 24", "gamerule keepInventory false"]


class Rewarder:
    def __init__(self, n):
        self.prev = [None] * n
        self.visited = [set() for _ in range(n)]

    def __call__(self, b, obs, w):
        """Reward for body b since its previous observation; returns (reward, done)."""
        if obs is None:
            return 0.0, False
        s = obs["stats"]
        p = self.prev[b]
        self.prev[b] = s
        cell = (int(obs["pos"][0] // 4), int(obs["pos"][2] // 4))
        new_cell = cell not in self.visited[b]
        self.visited[b].add(cell)
        if p is None:
            return 0.0, False
        d = {k: s[k] - p[k] for k in s}
        died = d["deaths"] > 0
        if died:
            self.visited[b].clear()
        r = (w["alive"] * (0 if obs["dead"] else 1) + w["explore"] * new_cell
             + w["collected"] * max(0, d["collected"] - d["logs"]) + w["logs"] * d["logs"]
             + w["mined"] * d["mined"] + w["eaten"] * d["eaten"] + w["damage"] * d["damage"] + w["death"] * died)
        return float(r), died


class PPO:
    def __init__(self, agent, lr=3e-4, horizon=128, epochs=4, minibatch=256, gamma=0.99, lam=0.95, clip=0.2,
                 ent_coef=0.003, vf_coef=0.5):
        self.agent = agent
        self.opt = torch.optim.Adam(agent.readout.parameters(), lr=lr)
        self.h, self.epochs, self.mb = horizon, epochs, minibatch
        self.gamma, self.lam, self.clip, self.ent_coef, self.vf_coef = gamma, lam, clip, ent_coef, vf_coef
        self.clear()

    def clear(self):
        self.buf = {k: [] for k in ("features", "task", "reflex", "acts", "logp", "value", "reward", "done", "mask")}

    def add(self, out, rewards, dones, mask):
        dev = self.agent.device
        self.buf["features"].append(out["features"].half())
        self.buf["task"].append(out["task"].clone())
        self.buf["reflex"].append({k: v.clone() for k, v in out["reflex"].items()})
        self.buf["acts"].append({k: v.clone() for k, v in out["acts"].items()})
        self.buf["logp"].append(out["logp"].clone())
        self.buf["value"].append(out["value"].clone())
        self.buf["reward"].append(torch.tensor(rewards, device=dev))
        self.buf["done"].append(torch.tensor(dones, device=dev, dtype=torch.float32))
        self.buf["mask"].append(torch.tensor(mask, device=dev, dtype=torch.float32))

    def ready(self):
        return len(self.buf["reward"]) > self.h

    def update(self):
        """Uses the first `horizon` steps; step horizon+1 provides the bootstrap value."""
        H = self.h
        B = self.buf
        values = torch.stack(B["value"][: H + 1])        # [H+1, n]
        rewards = torch.stack(B["reward"][1 : H + 1])    # reward observed after acting at step t
        dones = torch.stack(B["done"][1 : H + 1])
        mask = torch.stack(B["mask"][:H])
        adv = torch.zeros_like(rewards)
        last = torch.zeros_like(rewards[0])
        for t in reversed(range(H)):
            nonterm = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * nonterm - values[t]
            last = delta + self.gamma * self.lam * nonterm * last
            adv[t] = last
        ret = adv + values[:H]

        feats = torch.stack(B["features"][:H]).float().flatten(0, 1)
        task = torch.stack(B["task"][:H]).flatten(0, 1)
        reflex = {k: torch.stack([r[k] for r in B["reflex"][:H]]).flatten(0, 1) for k in HEADS}
        acts = {k: torch.stack([a[k] for a in B["acts"][:H]]).flatten(0, 1) for k in HEADS}
        old_logp = torch.stack(B["logp"][:H]).flatten()
        adv, ret, m = adv.flatten(), ret.flatten(), mask.flatten()
        keep = m > 0
        idx_all = torch.nonzero(keep).squeeze(1)
        a_k = adv[idx_all]
        adv[idx_all] = (a_k - a_k.mean()) / (a_k.std() + 1e-8)

        stats = []
        self.agent.readout.train()
        for _ in range(self.epochs):
            perm = idx_all[torch.randperm(len(idx_all), device=idx_all.device)]
            for i in range(0, len(perm), self.mb):
                j = perm[i : i + self.mb]
                logits, v = self.agent.readout(feats[j], {k: reflex[k][j] for k in HEADS}, task[j])
                logp, ent = evaluate(logits, {k: acts[k][j] for k in HEADS})
                ratio = torch.exp(logp - old_logp[j])
                pg = -torch.min(ratio * adv[j], ratio.clamp(1 - self.clip, 1 + self.clip) * adv[j]).mean()
                vf = (v - ret[j]).pow(2).mean()
                loss = pg + self.vf_coef * vf - self.ent_coef * ent.mean()
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.readout.parameters(), 0.5)
                self.opt.step()
                stats.append((pg.item(), vf.item(), ent.mean().item(), ((ratio - 1).abs() > self.clip).float().mean().item()))
        self.agent.readout.eval()
        # keep the bootstrap step as the start of the next rollout
        for k in self.buf:
            self.buf[k] = self.buf[k][H:]
        s = np.array(stats).mean(0)
        return {"policy_loss": s[0], "value_loss": s[1], "entropy": s[2], "clip_frac": s[3],
                "reward_per_step": float((rewards.flatten()[keep]).mean())}


async def main_async(args):
    run_dir = RUNS / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    rcon = Rcon(password=args.rcon_password)
    for c in WORLD_SETUP:
        rcon.cmd(c)

    server = BodyServer(args.bodies, port=args.ws_port)
    await server.start()
    agent = Agent(args.bodies, window_ms=args.window_ms, shuffled=args.shuffled)
    ppo = PPO(agent, lr=args.lr, horizon=args.horizon)
    update, stage = 0, ("speedrun" if args.task == "speedrun" else args.stage)
    if (run_dir / "latest.pt").exists() and not args.fresh:
        ck = torch.load(run_dir / "latest.pt", map_location=agent.device, weights_only=False)
        agent.readout.load_state_dict(ck["readout"])
        ppo.opt.load_state_dict(ck["opt"])
        update, stage = ck["update"], ck.get("stage", stage)
        print(f"resumed {run_dir / 'latest.pt'} at update {update}, stage {stage}")
    dash = Dashboard(agent, server, port=args.dashboard_port)
    await dash.start()

    def set_stage(name):
        for c in STAGES[name]["commands"]:
            rcon.cmd(c)
        print(f"== curriculum stage: {name}")

    set_stage(stage)
    speedrun = args.task == "speedrun"
    agent.episode_limit_s = args.episode_seconds
    rewarder = SpeedrunRewarder(args.bodies) if speedrun else Rewarder(args.bodies)
    ep_return = [0.0] * args.bodies
    ep_steps = [0] * args.bodies
    finished = []
    stage_scores = []
    best_splits = {}
    log = open(run_dir / "metrics.jsonl", "a")
    splits_log = open(run_dir / "splits.jsonl", "a")
    split_path = run_dir / "best_splits.json"
    if split_path.exists():
        best_splits = json.loads(split_path.read_text())

    async def reset_body(b):
        name = server.names[b]
        if name:
            rcon.cmd(f"clear {name}")
            rcon.cmd(f"effect clear {name}")
            rcon.cmd(f"spreadplayers {args.spawn_x} {args.spawn_z} 4 {args.spread} false {name}")
        await server.send(b, {"type": "reset"})

    started = [False] * args.bodies
    dopa_sums, dopa_n = {}, 0

    async def on_cycle(obs_list, out):
        nonlocal update, stage, dopa_n
        rewards, dones, mask = [], [], []
        for b, o in enumerate(obs_list):
            if speedrun and o is not None and not started[b]:
                started[b] = True
                await reset_body(b)
            if speedrun and rewarder.needs_resend(b):
                rewarder.awaiting[b] = 1
                await reset_body(b)
            if speedrun:
                r, died, new, comps = rewarder(b, o)
                if comps:
                    dopa_n += 1
                    for k, v in comps.items():
                        dopa_sums[k] = dopa_sums.get(k, 0.0) + v
                for name, split in new:
                    print(f"[{server.names[b]}] split {name} at {split:.1f}s")
                    if split < best_splits.get(name, float("inf")):
                        best_splits[name] = split
                        split_path.write_text(json.dumps(best_splits, indent=1))
                timeout = (o is not None and not rewarder.awaiting[b]
                           and o.get("episode_s", 0) >= args.episode_seconds)
                complete = rewarder.complete(b)
            else:
                r, died = rewarder(b, o, STAGES[stage]["w"])
                ep_steps[b] += o is not None
                timeout = ep_steps[b] >= args.episode_steps
                complete = False
            done = died or timeout or complete
            ep_return[b] += r
            if done:
                finished.append(ep_return[b])
                if speedrun and o is not None:
                    splits_log.write(json.dumps({"time": time.time(), "update": update, "body": b, "return": ep_return[b],
                                                 "splits": o.get("progress", {}), "died": died, "complete": complete}) + "\n")
                    splits_log.flush()
                    rewarder.reset(b)
                ep_return[b], ep_steps[b] = 0.0, 0
                if timeout or complete or (speedrun and died):
                    await reset_body(b)
                    if not speedrun:
                        rewarder.visited[b].clear()
            rewards.append(r)
            dones.append(float(done))
            mask.append(float(o is not None and not o.get("dead")))
        if speedrun:
            agent.dopamine = rewarder.drive()
        ppo.add(out, rewards, dones, mask)
        if not ppo.ready():
            return
        stats = await asyncio.get_running_loop().run_in_executor(None, ppo.update)
        update += 1
        recent = finished[-20:]
        stats.update(update=update, stage=stage, time=time.time(),
                     episodes=len(finished), mean_episode_return=float(np.mean(recent)) if recent else None,
                     reflex_agreement=float(out["reflex_agree"].mean()))
        if speedrun and dopa_n:
            stats.update({f"dopa_{k}": v / dopa_n for k, v in dopa_sums.items()})
            dopa_sums.clear()
            dopa_n = 0
        if speedrun:
            stats["best_splits"] = {k: round(v, 1) for k, v in best_splits.items()}
        stage_scores.append(stats["reward_per_step"])
        log.write(json.dumps(stats) + "\n")
        log.flush()
        dash.train = stats
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items() if k != "time"))
        ck = {"readout": agent.readout.state_dict(), "opt": ppo.opt.state_dict(), "update": update, "stage": stage,
              "args": vars(args)}
        torch.save(ck, run_dir / "latest.pt")
        if update % args.save_every == 0:
            torch.save(ck, run_dir / f"readout_{update:05d}.pt")
        i = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else len(STAGE_ORDER)
        if i + 1 < len(STAGE_ORDER) and len(stage_scores) >= args.stage_updates:
            stage = STAGE_ORDER[i + 1]
            stage_scores.clear()
            set_stage(stage)

    await control_loop(agent, server, dash, hz=args.hz, on_cycle=on_cycle,
                       stop=lambda: update >= args.updates)
    print("training finished")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["speedrun", "survival"], default="speedrun")
    ap.add_argument("--episode-seconds", type=float, default=900.0, help="speedrun: wall-clock time limit per attempt")
    ap.add_argument("--bodies", type=int, default=4)
    ap.add_argument("--run", default="default")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint in the run directory")
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--horizon", type=int, default=128, help="control cycles per PPO update")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--stage", choices=STAGE_ORDER, default="explore")
    ap.add_argument("--stage-updates", type=int, default=150, help="updates per curriculum stage")
    ap.add_argument("--episode-steps", type=int, default=3000, help="control cycles before a body is re-spread")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--shuffled", action="store_true")
    ap.add_argument("--window-ms", type=float, default=100.0)
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--spawn-x", type=int, default=591, help="forest near spawn for seed 20260903")
    ap.add_argument("--spawn-z", type=int, default=670)
    ap.add_argument("--spread", type=int, default=48)
    ap.add_argument("--rcon-password", default="flybrain")
    ap.add_argument("--ws-port", type=int, default=8765)
    ap.add_argument("--dashboard-port", type=int, default=8000)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
