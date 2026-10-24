"""Experiment-seven paths, provenance, and isolated writable state."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
DATA = ROOT / "data" / "exp007_psl_branch"
LEGACY = ROOT / "experiments" / "exp006_background_residual"
LEGACY_DATA = ROOT / "data" / "exp006_background_residual"
REPLACEMENTS = ROOT / "data" / "exp007_replacement_candidates_20261023"
DEFAULT_MODEL = ROOT.parent / "pyronear_lab" / "models" / "v0.6.0" / "model.zip"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def protocol():
    cfg = read_json(EXPERIMENT / "protocol.json")
    if cfg["projection_dimension"] < 2 * cfg["prototypes_per_class"]:
        raise ValueError("PUO requires projection dimension >= total prototype count")
    return cfg


def configure_cache():
    base = ROOT / ".cache" / "exp007"
    for key, name in (("TEMP", "temp"), ("TMP", "temp"), ("YOLO_CONFIG_DIR", "ultralytics"),
                      ("TORCH_HOME", "torch"), ("HF_HOME", "huggingface"), ("MPLCONFIGDIR", "matplotlib")):
        path = base / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    tempfile.tempdir = str(base / "temp")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def manifest():
    return read_json(DATA / "manifest.json")


def labels():
    return read_json(DATA / "labels.json")


def fingerprint(split, annotations=None):
    annotations = annotations if annotations is not None else labels()
    ids = [s["id"] for s in manifest()["sequences"] if s["split"] == split]
    return object_digest({i: annotations["sequences"].get(i, {}) for i in ids})


def code_fingerprints():
    paths = list((EXPERIMENT / "scripts").glob("*.py"))
    paths += list((EXPERIMENT / "scripts").glob("*.html"))
    paths += list((EXPERIMENT / "scripts").glob("*.ps1"))
    paths += [LEGACY / "scripts" / name for name in
              ("features.py", "common.py", "learning.py", "evaluation.py", "review.py", "review_assist.py", "review.html")]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in sorted(paths)}


def folder(args):
    return EXPERIMENT / "outputs" / args.run_id / args.setting


def require_cache(args):
    directory = folder(args)
    meta = read_json(directory / "run.json")
    if not meta.get("complete") or meta.get("partial"):
        raise ValueError("Run Extract/Setup to completion before training or evaluation")
    if meta["protocol_sha256"] != digest(EXPERIMENT / "protocol.json") or \
       meta["manifest_selection_sha256"] != digest(EXPERIMENT / "data_selection.json"):
        raise ValueError("Protocol or data selection changed; use a new RunId")
    if meta["extractor_sha256"] != digest(LEGACY / "scripts" / "features.py"):
        raise ValueError("Upstream feature extractor changed; create a new verified cache")
    return meta


def load_rows(args):
    meta = require_cache(args)
    rows = []
    for seq in meta["sequence_ids"]:
        path = folder(args) / f"{seq}.jsonl"
        if digest(path) != meta["record_sha256"][seq]:
            raise ValueError(f"Candidate records changed: {seq}")
        rows.extend(read_jsonl(path))
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate candidate IDs")
    sources = folder(args) / "feature_sources.json"
    if digest(sources) != meta["feature_sources_sha256"]:
        raise ValueError("Feature source inventory changed")
    for path, expected in read_json(sources).items():
        if digest(path) != expected:
            raise ValueError(f"Frozen feature bytes changed: {path}")
    return rows


def assert_training_identity(state, args):
    if state["protocol_sha256"] != digest(EXPERIMENT / "protocol.json") or \
       state["manifest_selection_sha256"] != digest(EXPERIMENT / "data_selection.json"):
        raise ValueError("Training protocol/data changed; use a new RunId")
    if state["code_fingerprints"] != code_fingerprints():
        raise ValueError("Training code changed since checkpoint creation; use a new RunId")
    if state["label_fingerprints"] != {s: fingerprint(s) for s in ("train", "val")}:
        raise ValueError("Train/val labels changed after training; use a new RunId")
    if state["cache_identity"] != object_digest(require_cache(args)):
        raise ValueError("Candidate cache changed after training")
