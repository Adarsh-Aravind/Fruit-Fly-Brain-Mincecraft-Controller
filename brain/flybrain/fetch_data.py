"""Download the MaleCNS v1.0 flat-connectome files (CC-BY 4.0, Berg et al., Cell 2026)."""
import shutil
import urllib.request

from .paths import ANNOTATIONS, DATA, FLAT_URL, NEUROTRANSMITTERS, WEIGHTS


def fetch(path):
    if path.exists():
        print(f"have {path.name} ({path.stat().st_size / 1e6:.0f} MB)")
        return
    tmp = path.with_suffix(".part")
    print(f"downloading {path.name} ...")
    with urllib.request.urlopen(FLAT_URL + path.name) as r, open(tmp, "wb") as out:
        shutil.copyfileobj(r, out, length=1 << 20)
    tmp.rename(path)


def main():
    DATA.mkdir(exist_ok=True)
    for p in (ANNOTATIONS, NEUROTRANSMITTERS, WEIGHTS):
        fetch(p)


if __name__ == "__main__":
    main()
