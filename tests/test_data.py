import io
import json
import tarfile

import pytest

from massive_es_ar.data import MEMBER, check_counts, extract_member, load


def _row(partition: str) -> dict:
    return {"partition": partition, "utt": "hola", "annot_utt": "hola"}


def _tarball(tmp_path, rows: list[dict]):
    payload = "".join(json.dumps(r) + "\n" for r in rows).encode()
    path = tmp_path / "massive.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for name, data in [(MEMBER, payload), ("1.1/data/en-US.jsonl", b"{}\n")]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def test_extract_keeps_only_es_es(tmp_path):
    rows = [_row("train"), _row("train"), _row("test")]
    dest = extract_member(_tarball(tmp_path, rows), tmp_path / "raw" / "es-ES.jsonl")
    assert list(dest.parent.iterdir()) == [dest]
    assert load(dest) == rows
    assert load(dest, "test") == [_row("test")]


def test_check_counts():
    rows = [_row("train")] * 2 + [_row("test")]
    assert check_counts(rows, {"train": 2, "test": 1}) == {"train": 2, "test": 1}
    with pytest.raises(ValueError, match="partition counts"):
        check_counts(rows, {"train": 3, "test": 1})
