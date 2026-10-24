"""Small shared helpers; all experiment writes stay under D:/AIC/AIC-seek."""
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
DATA = ROOT / "data" / "exp006_background_residual"
DEFAULT_MODEL = Path(r"D:\AIC\pyronear_lab\models\v0.6.0\model.zip")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def protocol():
    return read_json(EXPERIMENT / "protocol.json")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def configure_cache():
    cache = ROOT / ".cache" / "exp006"
    for key, name in (("TEMP", "temp"), ("TMP", "temp"), ("YOLO_CONFIG_DIR", "ultralytics"),
                      ("TORCH_HOME", "torch"), ("HF_HOME", "huggingface"), ("MPLCONFIGDIR", "matplotlib")):
        folder = cache / name
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    tempfile.tempdir = str(cache / "temp")
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


def load_manifest():
    path = DATA / "manifest.json"
    if not path.exists():
        selection = read_json(EXPERIMENT / "data_selection.json")
        write_json(path, {"created_at": now(), "download_status": "not_downloaded",
                         "sequences": selection["sequences"],
                         "quality_exclusions": selection.get("quality_exclusions", [])})
    return read_json(path)


def check_split_integrity(sequences):
    site_splits, event_splits = {}, {}
    for seq in sequences:
        split = seq["split"]
        site = seq["site"]
        if site in site_splits and site_splits[site] != split:
            raise ValueError(f"Site leakage: {site}")
        site_splits[site] = split
        event = seq.get("event_group")
        if event:
            if event in event_splits and event_splits[event] != split:
                raise ValueError(f"Event leakage: {event}")
            event_splits[event] = split
    return site_splits


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def append_jsonl(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def labels_digest():
    return digest(DATA / "labels.json")
