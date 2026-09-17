"""Batched leaky integrate-and-fire simulation of the whole connectome on the GPU.

Equations and constants follow Shiu et al. (Nature 2024):
    dv/dt = (v_rest - v + g) / tau_m        dg/dt = -g / tau_syn
    spike when v > v_thresh -> v = v_reset, refractory, and after the synaptic
    delay every postsynaptic g += w_syn * gain * (signed synapse count)
Sensory neurons are driven by independent Poisson spike trains.

State tensors are [B, N]: B bodies (Minecraft bots) share one wiring diagram.

Speed trick (exact, not an approximation): a spike emitted at step t only reaches
its targets at t + D (D = delay / dt = 18 steps). So we integrate D steps of
membrane dynamics using already-buffered input, then deliver all of the chunk's
spikes to the delay buffer in one event-driven scatter. Only the ~1% of neurons
that fire cost anything, instead of multiplying through all 6.2M synapses per step.
"""
import argparse
import time
from dataclasses import dataclass

import torch

from .connectome import Connectome


@dataclass
class LIFParams:
    dt: float = 0.1            # ms
    tau_m: float = 20.0        # ms
    tau_syn: float = 5.0       # ms
    v_rest: float = -52.0      # mV
    v_reset: float = -52.0     # mV
    v_thresh: float = -45.0    # mV
    t_ref: float = 2.2         # ms
    delay: float = 1.8         # ms
    w_syn: float = 0.275       # mV per synapse
    gain: float = 0.65         # global gain (keeps the male CNS out of runaway excitation)
    w_poisson: float = 68.75   # mV per Poisson input spike (250 * w_syn)
    kc_input_gain: float = 0.25  # synapses onto Kenyon cells; restores sparse mushroom-body coding
                                 # (~6% of KCs active instead of ~70%), as in fly-brain-minecraft


class Brain:
    def __init__(self, conn: Connectome, batch=1, params=LIFParams(), device="cuda", seed=0):
        self.p = params
        self.n = conn.n
        self.b = batch
        self.device = device
        # Out-edge table (CSC-like): edges sorted by presynaptic neuron.
        order = torch.argsort(conn.pre.long() * conn.n + conn.post.long())
        pre = conn.pre.long()[order]
        deg = torch.bincount(pre, minlength=self.n)
        start = torch.zeros(self.n, dtype=torch.long)
        start[1:] = torch.cumsum(deg, 0)[:-1]
        self.out_post = conn.post.long()[order].to(device)
        w = conn.weight[order].float() * params.w_syn * params.gain
        post = conn.post.long()[order]
        kc = torch.from_numpy(conn.cls == "Kenyon_Cell")
        w = torch.where(kc[post], w * params.kc_input_gain, w)
        self.out_w = w.to(device)
        self.out_deg = deg.to(device)
        self.out_start = start.to(device)

        self.D = max(1, round(params.delay / params.dt))
        self.ref_steps = round(params.t_ref / params.dt)
        self.k_m = params.dt / params.tau_m
        self.k_g = float(torch.exp(torch.tensor(-params.dt / params.tau_syn)))
        torch.manual_seed(seed)
        B, N, d = self.b, self.n, device
        self.v = torch.full((B, N), params.v_rest, device=d)
        self.g = torch.zeros(B, N, device=d)
        self.ref = torch.zeros(B, N, device=d)
        self.cur = torch.zeros(self.D, B, N, device=d)  # synaptic input arriving this chunk
        self.nxt = torch.zeros(self.D, B, N, device=d)  # ... and next chunk
        self.spk = torch.zeros(self.D, B, N, device=d)
        self.p_in = torch.zeros(B, N, device=d)          # Poisson spike probability per step
        self.counts = torch.zeros(B, N, device=d)
        self.use_graph = device == "cuda"
        self._graph = None

    def reset(self, which=None):
        """Silence all bodies (which=None) or the given batch indices."""
        sl = slice(None) if which is None else which
        self.v[sl] = self.p.v_rest
        for x in (self.g, self.ref, self.p_in):
            x[sl] = 0
        for x in (self.cur, self.nxt, self.spk):
            x[:, sl] = 0

    def set_input(self, rate_hz):
        """rate_hz: [B, N] Poisson input rates in Hz (zero for non-sensory neurons)."""
        self.p_in.copy_((rate_hz * self.p.dt * 1e-3).clamp(0, 1))

    @torch.no_grad()
    def run(self, ms):
        """Advance about `ms` of neural time (whole delay chunks); returns spike counts [B, N]."""
        chunks = max(1, round(ms / (self.D * self.p.dt)))
        counts = torch.zeros(self.b, self.n, device=self.device)
        for _ in range(chunks):
            self._integrate()
            counts += self.counts
            self.cur.copy_(self.nxt)
            self.nxt.zero_()
            self._deliver()
        return counts

    def _integrate(self):
        if not self.use_graph:
            return self._integrate_eager()
        if self._graph is None:
            self._capture()
        self._graph.replay()

    def _capture(self):
        saved = [x.clone() for x in (self.v, self.g, self.ref, self.spk, self.counts)]
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3):
                self._integrate_eager()
        torch.cuda.current_stream().wait_stream(s)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._integrate_eager()
        for x, y in zip((self.v, self.g, self.ref, self.spk, self.counts), saved):
            x.copy_(y)

    def _integrate_eager(self):
        """D steps of membrane dynamics with input already in the delay buffer (no host syncs)."""
        p = self.p
        for k in range(self.D):
            self.g.mul_(self.k_g).add_(self.cur[k])
            self.g.add_((torch.rand_like(self.p_in) < self.p_in).float(), alpha=p.w_poisson)
            self.v.add_((p.v_rest - self.v + self.g) * self.k_m * (self.ref <= 0))
            s = self.spk[k]
            s.copy_(self.v > p.v_thresh)
            self.v.sub_(s * (self.v - p.v_reset))
            self.ref.sub_(1).clamp_(min=0).add_(s * self.ref_steps)
        torch.sum(self.spk, 0, out=self.counts)

    def _deliver(self):
        """Event-driven: route this chunk's spikes (step k) to targets at step k of the next chunk."""
        B, N = self.b, self.n
        idx = self.spk.view(-1).nonzero().squeeze(1)  # flat over [D, B, N]
        if idx.numel() == 0:
            return
        kb = idx // N  # k * B + b
        n_i = idx % N
        lens = self.out_deg[n_i]
        total = int(lens.sum())
        if total == 0:
            return
        spike_of_edge = torch.repeat_interleave(torch.arange(idx.numel(), device=self.device), lens, output_size=total)
        first = torch.cumsum(lens, 0) - lens
        e = self.out_start[n_i][spike_of_edge] + torch.arange(total, device=self.device) - first[spike_of_edge]
        self.nxt.view(-1).index_add_(0, kb[spike_of_edge] * N + self.out_post[e], self.out_w[e])


def benchmark():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--ms", type=float, default=500)
    ap.add_argument("--rate", type=float, default=50)
    args = ap.parse_args()
    conn = Connectome.load()
    sensory = conn.select(types=["R1-R6"])
    for B in args.batch:
        brain = Brain(conn, batch=B)
        rate = torch.zeros(B, conn.n, device="cuda")
        rate[:, sensory] = args.rate
        brain.set_input(rate)
        brain.run(20)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        counts = brain.run(args.ms)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        active = (counts > 0).float().mean().item()
        print(f"B={B}: {args.ms / dt / 1000:.2f}x real time per body ({B * args.ms / dt / 1000:.2f}x total), "
              f"{active:.1%} of neurons spiked, mean rate {counts.mean().item() / args.ms * 1000:.2f} Hz")


if __name__ == "__main__":
    benchmark()
