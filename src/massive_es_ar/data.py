# %% [markdown]
# # MASSIVE 1.1, es-ES
#
# Downloads the MASSIVE 1.1 tarball, keeps only `1.1/data/es-ES.jsonl` under `data/raw/`,
# and checks the row count of each partition. Run with `uv run python -m massive_es_ar.data`.
#
# Not `datasets.load_dataset`: the Hub copy (`AmazonScience/massive`) is a loading script with
# no Parquet export, and `datasets` 4.0 removed script support.

# %%
import json
import shutil
import tarfile
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path

URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
MEMBER = "1.1/data/es-ES.jsonl"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
ES_ES = RAW_DIR / "es-ES.jsonl"

# Counted 2026-10-08
EXPECTED = {"train": 11_514, "dev": 2_033, "test": 2_974}


# %%
def extract_member(tarball: Path, dest: Path, member: str = MEMBER) -> Path:
    """Copy one file out of the tarball to `dest`, without unpacking the rest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball) as tar:
        src = tar.extractfile(member)
        if src is None:
            raise FileNotFoundError(f"{member} is not a regular file in {tarball}")
        with src, dest.open("wb") as out:
            shutil.copyfileobj(src, out)
    return dest


def download(dest: Path = ES_ES, url: str = URL) -> Path:
    """Fetch the tarball to a temp file and keep only the es-ES file. Skips if `dest` exists."""
    if dest.exists():
        return dest
    with tempfile.TemporaryDirectory() as tmp:
        tarball = Path(tmp) / "massive.tar.gz"
        with urllib.request.urlopen(url) as resp, tarball.open("wb") as out:
            shutil.copyfileobj(resp, out)
        return extract_member(tarball, dest)


def load(path: Path = ES_ES, partition: str | None = None) -> list[dict]:
    """Rows of the es-ES file, optionally only one partition (train, dev or test)."""
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if partition is not None:
        rows = [r for r in rows if r["partition"] == partition]
    return rows


def check_counts(rows: list[dict], expected: dict[str, int] = EXPECTED) -> Counter[str]:
    """Raise if the rows per partition differ from `expected`."""
    counts = Counter(r["partition"] for r in rows)
    if dict(counts) != expected:
        raise ValueError(f"partition counts {dict(counts)} != expected {expected}")
    return counts


# %%
if __name__ == "__main__":
    path = download()
    counts = check_counts(load(path))
    print(f"{path}: {dict(counts)}")
