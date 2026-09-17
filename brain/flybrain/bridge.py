"""Closed loop: Minecraft bodies <-> WebSocket <-> connectome brain <-> actions.

One control cycle (default 10 Hz):
  latest observation of every body -> sensory Poisson rates -> `window_ms` of LIF
  simulation -> descending/motor neuron rates -> reflex logits (+ learned readout)
  -> sampled actions -> sent back to each body.
"""
import asyncio
import json
import time
from collections import deque

import numpy as np
import torch
import websockets
from aiohttp import web

from .connectome import Connectome
from .lif import Brain
from .motor import HEADS, MotorMap
from .policy import Readout, sample, to_commands
from .senses import SensoryMap
from .task import N_TASK, batch_task

DASHBOARD_HTML = (__import__("pathlib").Path(__file__).parent / "dashboard.html")


class BodyServer:
    """Bots connect here; we keep the latest observation from each."""

    def __init__(self, n_bodies, host="127.0.0.1", port=8765):
        self.n = n_bodies
        self.host, self.port = host, port
        self.sockets = [None] * n_bodies
        self.names = [None] * n_bodies
        self.obs = [None] * n_bodies
        self.obs_time = [0.0] * n_bodies

    async def start(self):
        # No keepalive pings: bodies are local, and a busy GPU cycle would otherwise time them out.
        self.server = await websockets.serve(self._handle, self.host, self.port, max_size=2**22, ping_interval=None)
        print(f"brain listening for bodies on ws://{self.host}:{self.port} ({self.n} slots)")

    async def _handle(self, ws):
        slot = None
        try:
            async for raw in ws:
                msg = json.loads(raw)
                if msg["type"] == "hello":
                    slot = int(msg["id"])
                    if slot >= self.n:
                        print(f"body {msg['name']} has id {slot} but only {self.n} slots; ignoring")
                        return
                    old = self.sockets[slot]
                    if old is not None and old is not ws:
                        print(f"body {slot} reconnected; closing its previous stream")
                        await old.close()
                    self.sockets[slot], self.names[slot] = ws, msg["name"]
                    print(f"body {slot} connected: {msg['name']}")
                elif msg["type"] == "obs" and slot is not None:
                    self.obs[slot] = msg["obs"]
                    self.obs_time[slot] = time.time()
        finally:
            if slot is not None and slot < self.n and self.sockets[slot] is ws:
                self.sockets[slot] = None
                self.obs[slot] = None
                print(f"body {slot} disconnected")

    def connected(self):
        return [s is not None and o is not None for s, o in zip(self.sockets, self.obs)]

    async def send(self, slot, msg):
        ws = self.sockets[slot]
        if ws is not None:
            try:
                await ws.send(json.dumps(msg))
            except websockets.ConnectionClosed:
                pass

    async def send_actions(self, commands):
        for ws, cmd in zip(self.sockets, commands):
            if ws is not None and cmd is not None:
                try:
                    await ws.send(json.dumps({"type": "action", "action": cmd}))
                except websockets.ConnectionClosed:
                    pass


class Agent:
    def __init__(self, n_bodies, window_ms=100.0, shuffled=False, pure_fly=False, device="cuda"):
        conn = Connectome.load()
        if shuffled:
            conn = conn.shuffled()
            print("WARNING: using a degree-preserving SHUFFLED connectome (control condition)")
        self.conn = conn
        self.device = device
        self.window_ms = window_ms
        self.pure_fly = pure_fly
        self.brain = Brain(conn, batch=n_bodies, device=device)
        self.senses = SensoryMap(conn)
        self.motor = MotorMap(conn)
        self.readout = Readout(len(self.motor.feature_idx), N_TASK).to(device)
        self.episode_limit_s = 900.0
        self.raster_idx = self._raster_neurons()
        self.was_dead = [False] * n_bodies
        self.last_counts = None
        self.dopamine = None  # [(pam, ppl1)] per body, set by the trainer
        print(f"brain: {conn.n} neurons, {len(conn.weight):,} synaptic edges, {n_bodies} bodies, "
              f"{len(self.motor.feature_idx)} readout neurons")

    def _raster_neurons(self):
        order = []
        for name in self.motor.names:
            order += list(self.motor.groups[name])
        return torch.as_tensor(order[:96], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def active_graph(self, body, max_edges=2500):
        """Neurons that spiked in the last cycle for one body, plus the strongest synapses that
        connected two of them (i.e. a spike that plausibly travelled pre -> post)."""
        counts = self.last_counts
        if counts is None:
            return np.zeros(0, np.uint32), np.zeros(0, np.uint8), np.zeros((0, 2), np.uint32), np.zeros(0, np.uint8)
        row = counts[body]
        idx = torch.nonzero(row).squeeze(1)
        n_spk = row[idx].clamp(max=255).to(torch.uint8)
        b = self.brain
        lens = b.out_deg[idx]
        total = int(lens.sum())
        pairs = torch.zeros(0, 2, dtype=torch.long, device=self.device)
        signs = torch.zeros(0, dtype=torch.uint8, device=self.device)
        if total:
            src = torch.repeat_interleave(torch.arange(len(idx), device=self.device), lens, output_size=total)
            first = torch.cumsum(lens, 0) - lens
            e = b.out_start[idx][src] + torch.arange(total, device=self.device) - first[src]
            post = b.out_post[e]
            live = row[post] > 0
            e, src, post = e[live], src[live], post[live]
            if len(e) > max_edges:
                top = torch.topk(b.out_w[e].abs(), max_edges).indices
                e, src, post = e[top], src[top], post[top]
            pairs = torch.stack([idx[src], post], 1)
            signs = (b.out_w[e] > 0).to(torch.uint8)
        return (idx.to(torch.int32).cpu().numpy().astype(np.uint32), n_spk.cpu().numpy(),
                pairs.to(torch.int32).cpu().numpy().astype(np.uint32), signs.cpu().numpy())

    @torch.no_grad()
    def think(self, obs_list, greedy=False):
        """Run one control cycle for all bodies. Returns a dict with everything downstream needs."""
        for b, o in enumerate(obs_list):
            dead = o is None or bool(o.get("dead"))
            if dead and not self.was_dead[b]:
                self.brain.reset([b])
            self.was_dead[b] = dead
        rates_in = self.senses.rates(obs_list, self.device, self.dopamine)
        self.brain.set_input(rates_in)
        counts = self.brain.run(self.window_ms)
        group_rates = self.motor.group_rates(counts, self.window_ms)
        task, craft_mask = batch_task(obs_list, self.device, self.episode_limit_s)
        alive = torch.tensor([o is not None and not o.get("dead") for o in obs_list], device=self.device)
        reflex = self.motor.reflex_logits(group_rates, craft_mask, alive)
        features = self.motor.features(counts, self.window_ms)
        if self.pure_fly:
            logits, value = reflex, torch.zeros(len(obs_list), device=self.device)
        else:
            logits, value = self.readout(features, reflex, task)
        self.last_counts = counts
        acts, logp, _ = sample(logits, greedy=greedy)
        reflex_choice = {k: reflex[k].argmax(-1) for k in HEADS}
        agree = torch.stack([(acts[k] == reflex_choice[k]).float() for k in HEADS], 1).mean(1)
        return {
            "features": features, "task": task, "reflex": reflex, "acts": acts, "logp": logp, "value": value,
            "group_rates": group_rates, "counts": counts, "rates_in": rates_in, "reflex_agree": agree,
        }


class Dashboard:
    """Tiny web page at http://localhost:8000 showing what the fly brain is doing."""

    def __init__(self, agent, server, port=8000):
        self.agent, self.server, self.port = agent, server, port
        self.state = {}
        self.raster = deque(maxlen=60)
        self.train = {}

    def update(self, out, obs_list, cmds, cycle_s):
        a = self.agent
        self.raster.append((out["counts"][0, a.raster_idx] > 0).to(torch.uint8).tolist())
        rin = out["rates_in"][0]
        bodies = []
        for b, o in enumerate(obs_list):
            if o is None:
                bodies.append(None)
                continue
            bodies.append({
                "name": self.server.names[b], "health": o["health"], "food": o["food"], "stats": o["stats"],
                "inventory": o["inventory"], "target": o.get("target"), "action": cmds[b],
                "reflex_agree": float(out["reflex_agree"][b]),
                "progress": o.get("progress", {}), "episode_s": o.get("episode_s"), "items": o.get("items", {}),
                "busy": o.get("busy", False),
                "dopamine": list(a.dopamine[b]) if a.dopamine is not None else [0.0, 0.0],
                "motor": {n: float(out["group_rates"][b, i]) for i, n in enumerate(a.motor.names)},
            })
        self.state = {
            "t": time.time(), "cycle_ms": cycle_s * 1000, "window_ms": a.window_ms,
            "realtime": a.window_ms / 1000 / max(cycle_s, 1e-6), "pure_fly": a.pure_fly,
            "sensory": {k: float(rin[torch.as_tensor(v, device=rin.device)].mean()) for k, v in a.senses.groups.items()},
            "active_frac": float((out["counts"] > 0).float().mean()),
            "bodies": bodies, "raster": list(self.raster),
            "raster_labels": self._raster_labels(), "train": self.train,
        }

    def _raster_labels(self):
        labels, n = [], 0
        for name in self.agent.motor.names:
            k = len(self.agent.motor.groups[name])
            labels.append([name, n, min(n + k, 96)])
            n += k
            if n >= 96:
                break
        return labels

    async def _brain_meta(self, request):
        from .brainmap import CLASS_ORDER, load
        if not hasattr(self, "_map"):
            self._map = await asyncio.get_running_loop().run_in_executor(None, load)
        a = self.agent
        groups = {**{f"in:{k}": v for k, v in a.senses.groups.items()}, **{f"out:{k}": v for k, v in a.motor.groups.items()}}
        return web.json_response({
            "n": a.conn.n, "edges": len(a.conn.weight), "classes": CLASS_ORDER,
            "class_counts": np.bincount(self._map[1], minlength=len(CLASS_ORDER)).tolist(),
            "bodies": a.brain.b, "groups": {k: np.asarray(v).tolist() for k, v in groups.items()},
            "cord": [i for i, sc in enumerate(a.conn.superclass) if sc.startswith("vnc_") or "ascending" in sc],
        })

    async def _brain_positions(self, request):
        from .brainmap import load
        if not hasattr(self, "_map"):
            self._map = await asyncio.get_running_loop().run_in_executor(None, load)
        return web.Response(body=self._map[0].tobytes() + self._map[1].tobytes(), content_type="application/octet-stream")

    async def _brain_activity(self, request):
        body = min(int(request.query.get("body", 0)), self.agent.brain.b - 1)
        idx, nspk, pairs, signs = await asyncio.get_running_loop().run_in_executor(None, self.agent.active_graph, body)
        header = np.array([len(idx), len(pairs)], np.uint32)
        payload = header.tobytes() + idx.tobytes() + pairs.tobytes() + nspk.tobytes() + signs.tobytes()
        return web.Response(body=payload, content_type="application/octet-stream")

    async def start(self):
        app = web.Application()
        app.router.add_get("/", lambda r: web.FileResponse(DASHBOARD_HTML))
        app.router.add_get("/brain", lambda r: web.FileResponse(DASHBOARD_HTML.with_name("brain.html")))
        app.router.add_get("/brain/meta", self._brain_meta)
        app.router.add_get("/brain/positions", self._brain_positions)
        app.router.add_get("/brain/activity", self._brain_activity)
        app.router.add_get("/state", lambda r: web.json_response(self.state))
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", self.port).start()
        print(f"dashboard on http://localhost:{self.port}")


async def control_loop(agent, server, dashboard=None, hz=10.0, greedy=False, on_cycle=None, stop=None):
    """Runs forever (or until stop() is true). on_cycle(obs_list, out) may be async and is used for training."""
    loop = asyncio.get_running_loop()
    period = 1.0 / hz
    while stop is None or not stop():
        t0 = time.perf_counter()
        live = server.connected()
        if not any(live):
            await asyncio.sleep(0.25)
            continue
        obs_list = [o if ok else None for o, ok in zip(server.obs, live)]
        out = await loop.run_in_executor(None, agent.think, obs_list, greedy)
        cmds = to_commands(out["acts"])
        cmds = [c if ok else None for c, ok in zip(cmds, live)]
        await server.send_actions(cmds)
        if on_cycle is not None:
            r = on_cycle(obs_list, out)
            if asyncio.iscoroutine(r):
                await r
        cycle = time.perf_counter() - t0
        if dashboard is not None:
            dashboard.update(out, obs_list, cmds, cycle)
        await asyncio.sleep(max(0.0, period - cycle))
