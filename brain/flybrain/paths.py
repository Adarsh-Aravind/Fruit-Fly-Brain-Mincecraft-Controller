from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RUNS = ROOT / "runs"

FLAT_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
ANNOTATIONS = DATA / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = DATA / "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS = DATA / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
GRAPH = DATA / "graph.pt"
