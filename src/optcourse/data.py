"""Explicit data preparation; byte windows never cross split boundaries."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def download_shakespeare(output):
    """Network access is confined to this explicitly invoked function."""
    output = Path(output)
    path = output / "shakespeare_input.txt"
    if path.exists():
        raise FileExistsError(f"Raw data already exists: {path}; use --input to reuse it")
    with urlopen(SHAKESPEARE_URL, timeout=60) as response:
        content = response.read()
    output.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def prepare_data(input_path, output, ratios=(.9, .05, .05), source_url=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite dataset directory: {output}")
    if len(ratios) != 3 or any(x <= 0 for x in ratios) or abs(sum(ratios) - 1) > 1e-8:
        raise ValueError("Three positive split ratios must sum to one")
    input_path = Path(input_path)
    content = input_path.read_bytes()
    end_train, end_val = int(len(content) * ratios[0]), int(len(content) * sum(ratios[:2]))
    parts = dict(train=content[:end_train], val=content[end_train:end_val], test=content[end_val:])
    if min(map(len, parts.values())) < 2:
        raise ValueError("Text is too short to create three nonempty next-byte splits")
    output.mkdir(parents=True)
    manifest = dict(schema_version=1, encoding="raw_utf8_bytes", vocabulary_size=256,
                    source_path=str(input_path.resolve()), source_url=source_url,
                    source_sha256=sha256(content), split_method="contiguous_no_overlap", splits={})
    for name, payload in parts.items():
        (output / f"{name}.bin").write_bytes(payload)
        manifest["splits"][name] = dict(bytes=len(payload), sha256=sha256(payload))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class ByteDataset:
    def __init__(self, directory):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("vocabulary_size") != 256 or self.manifest.get("schema_version") != 1:
            raise ValueError("Unsupported dataset format")
        self.parts = {}
        for name in ("train", "val", "test"):
            payload = (directory / f"{name}.bin").read_bytes()
            spec = self.manifest["splits"][name]
            if len(payload) != spec["bytes"] or sha256(payload) != spec["sha256"]:
                raise ValueError(f"Dataset checksum mismatch: {name}")
            self.parts[name] = torch.from_numpy(np.frombuffer(payload, dtype=np.uint8).copy())
        if sha256(b"".join(self.parts[name].numpy().tobytes() for name in ("train", "val", "test"))) != self.manifest["source_sha256"]:
            raise ValueError("Dataset source checksum mismatch")
        identity = {key: self.manifest[key] for key in
                    ("schema_version", "encoding", "vocabulary_size", "split_method", "source_sha256", "splits")}
        self.fingerprint = sha256(json.dumps(identity, sort_keys=True).encode("utf-8"))

    def batch(self, split, batch_size, length, generator, device):
        if split not in self.parts or batch_size < 1 or length < 1:
            raise ValueError("Invalid split, batch size or sequence length")
        data = self.parts[split]
        if len(data) <= length:
            raise ValueError(f"Split {split} has {len(data)} bytes; needs at least {length + 1}")
        starts = torch.randint(len(data) - length, (batch_size,), generator=generator)
        windows = torch.stack([data[i:i + length + 1] for i in starts.tolist()]).long()
        return windows[:, :-1].to(device), windows[:, 1:].to(device)
