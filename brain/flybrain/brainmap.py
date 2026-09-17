"""3D layout of every simulated neuron, for the live whole-brain view.

Neurons with an annotated soma (or tosoma) location use it (141k of 165k). The rest,
mostly sensory neurons whose cell bodies sit outside the CNS, are placed at the
synapse-weighted mean position of their placed partners.

    python -m flybrain.brainmap      # writes data/brainmap.npz
"""
import numpy as np
import pyarrow.feather as feather
import torch

from .connectome import Connectome
from .paths import ANNOTATIONS, DATA

BRAINMAP = DATA / "brainmap.npz"
CLASS_ORDER = [
    "sensory", "visual", "central", "optic_lobe", "vnc", "ascending", "descending", "motor", "other",
]


def class_of(superclass):
    s = superclass or ""
    if "sensory" in s:
        return "sensory"
    if s.startswith("visual"):
        return "visual"
    if s.startswith("cb_") and "motor" not in s:
        return "central"
    if s.startswith("ol_"):
        return "optic_lobe"
    if s.startswith("vnc_") and "motor" not in s:
        return "vnc"
    if "ascending" in s:
        return "ascending"
    if "descending" in s:
        return "descending"
    if "motor" in s or "efferent" in s:
        return "motor"
    return "other"


def build():
    conn = Connectome.load()
    ann = feather.read_table(ANNOTATIONS, columns=["bodyId", "somaLocation", "tosomaLocation"]).to_pandas()
    ann = ann.set_index("bodyId").reindex(conn.body_ids)
    pos = np.full((conn.n, 3), np.nan, np.float64)
    for col in ("tosomaLocation", "somaLocation"):
        v = ann[col].to_numpy()
        ok = np.array([x is not None and not (isinstance(x, float)) and len(x) == 3 for x in v])
        pos[ok] = np.stack(v[ok]).astype(np.float64)
    known = ~np.isnan(pos[:, 0])
    print(f"{known.sum()} neurons with annotated positions")

    pre = conn.pre.numpy().astype(np.int64)
    post = conn.post.numpy().astype(np.int64)
    w = np.abs(conn.weight.numpy()).astype(np.float64)
    rng = np.random.default_rng(0)
    for it in range(6):
        placed = ~np.isnan(pos[:, 0])
        acc = np.zeros((conn.n, 3))
        tot = np.zeros(conn.n)
        for a, b in ((pre, post), (post, pre)):  # neighbours in both directions
            m = placed[b] & ~placed[a]
            np.add.at(acc, a[m], pos[b[m]] * w[m, None])
            np.add.at(tot, a[m], w[m])
        new = (tot > 0) & ~placed
        pos[new] = acc[new] / tot[new, None]
        print(f"  pass {it}: placed {new.sum()} more")
        if not new.any():
            break
    missing = np.isnan(pos[:, 0])
    pos[missing] = np.nanmean(pos, 0)
    # jitter unplaced/derived points so they don't stack on one voxel
    derived = ~known
    pos[derived] += rng.normal(0, 800, (derived.sum(), 3))

    # centre, scale to ~[-1, 1], flip y (EM y grows ventrally)
    c = np.nanmedian(pos, 0)
    pos = (pos - c) / np.abs(pos - c).max()
    pos[:, 1] *= -1
    classes = np.array([CLASS_ORDER.index(class_of(s)) for s in conn.superclass], np.uint8)
    np.savez_compressed(BRAINMAP, pos=pos.astype(np.float32), classes=classes)
    print(f"saved {BRAINMAP}")


def load():
    if not BRAINMAP.exists():
        build()
    d = np.load(BRAINMAP)
    return d["pos"], d["classes"]


if __name__ == "__main__":
    build()
