"""Build and load the signed MaleCNS connectivity graph.

Neurons: every body with status == "Traced" (165,122 in v1.0).
Edges: pre -> post with >= MIN_SYNAPSES synapses, signed by the presynaptic
neuron's predicted neurotransmitter (GABA / glutamate inhibitory, everything else
excitatory), the same convention as Shiu et al., Nature 2024.
"""
import argparse
import re
from dataclasses import dataclass

import numpy as np
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import torch

from .paths import ANNOTATIONS, GRAPH, NEUROTRANSMITTERS, WEIGHTS

MIN_SYNAPSES = 5
INHIBITORY = {"gaba", "glutamate"}


def _side(row_soma, row_root):
    s = row_soma if isinstance(row_soma, str) and row_soma in ("L", "R", "M") else row_root
    return s if isinstance(s, str) else ""


def build(min_synapses=MIN_SYNAPSES):
    ann = feather.read_table(
        ANNOTATIONS,
        columns=["bodyId", "status", "type", "instance", "superclass", "class", "somaSide", "rootSide", "synonyms"],
    ).to_pandas()
    ann = ann[ann.status == "Traced"].sort_values("bodyId").reset_index(drop=True)
    ids = ann.bodyId.to_numpy(np.int64).copy()
    n = len(ids)
    print(f"{n} traced neurons")

    nt = feather.read_table(NEUROTRANSMITTERS, columns=["body", "consensus_nt", "predicted_nt"]).to_pandas()
    nt = nt.set_index("body")
    label = nt.consensus_nt.fillna(nt.predicted_nt).reindex(ids).fillna("unknown").str.lower()
    sign = np.where(label.isin(INHIBITORY).to_numpy(), -1.0, 1.0).astype(np.float32)
    print("neurotransmitters:", label.value_counts().to_dict())

    pre_all, post_all, w_all = [], [], []
    reader = ipc.open_file(WEIGHTS)
    kept = 0
    for i in range(reader.num_record_batches):
        b = reader.get_batch(i)
        w = b.column("weight").to_numpy()
        m = w >= min_synapses
        if not m.any():
            continue
        pre = b.column("body_pre").to_numpy()[m]
        post = b.column("body_post").to_numpy()[m]
        w = w[m]
        pi = np.searchsorted(ids, pre)
        qi = np.searchsorted(ids, post)
        pi = np.minimum(pi, n - 1)
        qi = np.minimum(qi, n - 1)
        ok = (ids[pi] == pre) & (ids[qi] == post) & (pi != qi)
        pre_all.append(pi[ok].astype(np.int32))
        post_all.append(qi[ok].astype(np.int32))
        w_all.append(w[ok].astype(np.float32))
        kept += int(ok.sum())
        if i % 200 == 0:
            print(f"  batch {i}/{reader.num_record_batches}: {kept:,} edges kept")
    pre = np.concatenate(pre_all)
    post = np.concatenate(post_all)
    w = np.concatenate(w_all) * sign[pre]
    print(f"{len(w):,} edges with >= {min_synapses} synapses")

    order = np.lexsort((pre, post))  # CSR rows = post, cols = pre
    torch.save(
        {
            "body_ids": torch.from_numpy(ids),
            "post": torch.from_numpy(post[order]),
            "pre": torch.from_numpy(pre[order]),
            "weight": torch.from_numpy(w[order]),
            "nt": label.tolist(),
            "type": ann["type"].fillna("").tolist(),
            "instance": ann["instance"].fillna("").tolist(),
            "superclass": ann["superclass"].fillna("").tolist(),
            "class": ann["class"].fillna("").tolist(),
            "side": [_side(s, r) for s, r in zip(ann.somaSide, ann.rootSide)],
            "synonyms": ann["synonyms"].fillna("").tolist(),
            "min_synapses": min_synapses,
        },
        GRAPH,
    )
    print(f"saved {GRAPH}")


@dataclass
class Connectome:
    body_ids: np.ndarray
    type: np.ndarray
    superclass: np.ndarray
    cls: np.ndarray
    side: np.ndarray
    synonyms: np.ndarray
    nt: np.ndarray
    post: torch.Tensor
    pre: torch.Tensor
    weight: torch.Tensor

    @property
    def n(self):
        return len(self.body_ids)

    @classmethod
    def load(cls, path=GRAPH):
        if not path.exists():
            raise SystemExit(f"{path} missing: run `python -m flybrain.connectome --build` first")
        g = torch.load(path, weights_only=False)
        return cls(
            body_ids=g["body_ids"].numpy(),
            type=np.array(g["type"], dtype=object),
            superclass=np.array(g["superclass"], dtype=object),
            cls=np.array(g["class"], dtype=object),
            side=np.array(g["side"], dtype=object),
            synonyms=np.array(g["synonyms"], dtype=object),
            nt=np.array(g["nt"], dtype=object),
            post=g["post"],
            pre=g["pre"],
            weight=g["weight"],
        )

    def select(self, types=None, pattern=None, superclass=None, cls=None, side=None):
        """Indices of neurons matching every given criterion."""
        m = np.ones(self.n, bool)
        if types is not None:
            m &= np.isin(self.type, list(types))
        if pattern is not None:
            rx = re.compile(pattern)
            m &= np.array([bool(rx.fullmatch(t)) for t in self.type])
        if superclass is not None:
            m &= np.isin(self.superclass, [superclass] if isinstance(superclass, str) else list(superclass))
        if cls is not None:
            m &= np.isin(self.cls, [cls] if isinstance(cls, str) else list(cls))
        if side is not None:
            m &= self.side == side
        return np.flatnonzero(m)

    def shuffled(self, seed=0):
        """Degree-preserving control: same out-edges per neuron, random targets."""
        g = torch.Generator().manual_seed(seed)
        post = self.post[torch.randperm(len(self.post), generator=g)]
        order = torch.argsort(post * self.n + self.pre)
        return Connectome(self.body_ids, self.type, self.superclass, self.cls, self.side, self.synonyms, self.nt,
                          post[order], self.pre[order], self.weight[order])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--min-synapses", type=int, default=MIN_SYNAPSES)
    args = ap.parse_args()
    if args.build or not GRAPH.exists():
        build(args.min_synapses)
    c = Connectome.load()
    print(f"{c.n} neurons, {len(c.weight):,} edges, {(c.weight < 0).float().mean():.1%} inhibitory")
