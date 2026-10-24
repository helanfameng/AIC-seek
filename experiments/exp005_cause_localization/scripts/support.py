"""Standard-library inputs, annotation semantics and experiment bookkeeping."""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
SETTINGS = ("full_baseline", "low_threshold", "high_resolution", "combined")
STAGES = ("crop", "oracle", "association", "calibration")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def default_run_id():
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")


def git_commit():
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def sigmoid(score):
    return 1 / (1 + math.exp(-score)) if score >= 0 else math.exp(score) / (1 + math.exp(score))


def calibrated(logit, span, confidence, n_tubes, calibrator):
    require(calibrator["features"] == ["logit", "log_len", "mean_conf", "n_tubes"], "Unexpected calibrator schema")
    values = (logit, math.log1p(span), confidence, n_tubes)
    score = calibrator["intercept"] + sum(w * v for w, v in zip(calibrator["coefficients"], values, strict=True))
    return sigmoid(score)


def norm_to_xyxy(box, width, height):
    cx, cy, w, h = box
    return [(cx - w / 2) * width, (cy - h / 2) * height,
            (cx + w / 2) * width, (cy + h / 2) * height]


def xyxy_to_norm(box, width, height):
    x0, y0, x1, y1 = box
    return [(x0 + x1) / (2 * width), (y0 + y1) / (2 * height),
            (x1 - x0) / width, (y1 - y0) / height]


def overlaps(reference, predicted):
    intersection = max(0, min(reference[2], predicted[2]) - max(reference[0], predicted[0])) * max(
        0, min(reference[3], predicted[3]) - max(reference[1], predicted[1]))
    area_ref = (reference[2] - reference[0]) * (reference[3] - reference[1])
    area_pred = (predicted[2] - predicted[0]) * (predicted[3] - predicted[1])
    require(area_ref > 0 and area_pred > 0, "Invalid geometry")
    return {"iou": intersection / (area_ref + area_pred - intersection),
            "reference_coverage": intersection / area_ref,
            "prediction_inside_reference": intersection / area_pred}


def union_box(boxes):
    require(bool(boxes), "No human boxes for this causal window")
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def empty_state(offset, first_box, last_box):
    if offset < 0:
        return "before_official_onset"
    if first_box is None:
        return "unresolved"
    if offset < first_box:
        return "not_discernible"
    if offset > last_box:
        return "visible_omitted_high_resolution_covered"
    return "unresolved"


class Inputs:
    def __init__(self):
        self.protocol = read_json(EXPERIMENT / "protocol.json")
        self.source_protocol = read_json(ROOT / self.protocol["source_protocol"])
        self.events = {e["key"]: e for e in self.source_protocol["events"]}
        self.frames, self.source_files, self.sources = {}, {}, {}
        for event, metadata in self.events.items():
            folder = ROOT / self.source_protocol["dataset_root_relative"] / metadata["event_id"]
            manifest_path = folder / "manifest.json"
            via_path = folder / self.protocol["via"][event]["filename"]
            require(sha256(via_path) == self.protocol["via"][event]["sha256"], f"VIA changed: {via_path}; update the experiment protocol explicitly")
            manifest, via = read_json(manifest_path), read_json(via_path)
            records = list(via["_via_img_metadata"].values())
            by_name = {r["filename"]: r for r in records}
            require(len(records) == len(by_name) == 81, f"VIA must contain 81 unique images: {event}")
            items = sorted(manifest["files"], key=lambda r: r["offset_seconds"])
            require([r["offset_seconds"] for r in items] == list(range(-2400, 2401, 60)), f"Incomplete manifest: {event}")
            require(set(by_name) == {r["filename"] for r in items}, f"VIA/manifest mismatch: {event}")
            box_offsets = [r["offset_seconds"] for r in items if by_name[r["filename"]]["regions"]]
            first, last = (min(box_offsets), max(box_offsets)) if box_offsets else (None, None)
            rows = []
            for index, item in enumerate(items, 1):
                require(item["timestamp"] - item["offset_seconds"] == metadata["origin_timestamp"], "Onset mismatch")
                boxes = []
                for region in by_name[item["filename"]]["regions"]:
                    shape = region["shape_attributes"]
                    require(shape["name"] == "rect", "Only a coarse rectangular plume extent is supported")
                    x, y, w, h = [shape[k] for k in ("x", "y", "width", "height")]
                    require(w > 0 and h > 0 and 0 <= x < x + w <= item["width"]
                            and 0 <= y < y + h <= item["height"], "Invalid human box")
                    boxes.append([x, y, x + w, y + h])
                require(len(boxes) <= 1 and (not boxes or item["offset_seconds"] >= 0), "Unexpected human region layout")
                rows.append({**item, "path": folder / item["filename"], "source_frame_1based": index,
                             "manual_box": boxes[0] if boxes else None,
                             "visibility": "visible_manual_box" if boxes else empty_state(item["offset_seconds"], first, last),
                             "human_onset_seconds": first})
            self.frames[event] = {Path(r["filename"]).stem: r for r in rows}
            self.source_files[event] = [manifest_path, via_path]

    def source(self, event, setting):
        key = event, setting
        if key not in self.sources:
            folder = ROOT / self.protocol["source_batch"] / event / setting
            run_path, steps_path = folder / "run.json", folder / "steps.jsonl"
            run = read_json(run_path)
            require(run["status"] == "completed", f"Incomplete exp004 source: {folder}")
            require(run["plan"]["model_sha256"] == self.protocol["model_sha256"], "Unexpected source model")
            steps = [json.loads(line) for line in steps_path.read_text(encoding="utf-8").splitlines()]
            require(len(steps) == 81, "Expected 81 source prediction steps")
            expected = list(self.frames[event].values())
            for row, step in zip(expected, steps, strict=True):
                require(row["filename"] == step["summary"]["filename"] and row["offset_seconds"] == step["summary"]["offset_seconds"], "Source step timeline mismatch")
                ids = step["effective_frame_ids"]
                require(ids[-1] == Path(row["filename"]).stem and len(ids) <= 20, "Invalid effective window")
                require(all(self.frames[event][fid]["offset_seconds"] <= row["offset_seconds"] for fid in ids), "Future image in source window")
            self.sources[key] = {"run": run, "steps": steps, "paths": [run_path, steps_path]}
        return self.sources[key]

    def verify_images(self, events):
        count = 0
        for event in sorted(set(events)):
            for row in self.frames[event].values():
                require(row["path"].stat().st_size == row["bytes"] and sha256(row["path"]) == row["sha256"], f"Original image changed: {row['path']}")
                count += 1
        return count

    def provenance(self, job):
        paths = [EXPERIMENT / "protocol.json", ROOT / self.protocol["source_protocol"], *self.source_files[job["event"]]]
        paths.extend(sorted((EXPERIMENT / "scripts").glob("*.py")))
        if job["stage"] != "oracle":
            paths.extend(self.source(job["event"], job["setting"])["paths"])
        return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


def select_steps(inputs, job, offset_min=None, offset_max=None, limit=None):
    if job["stage"] == "oracle":
        candidates = list(inputs.frames[job["event"]].values())
        candidates = [r for r in candidates if r["manual_box"]]
    else:
        candidates = inputs.source(job["event"], job["setting"])["steps"]
        if job["stage"] == "crop":
            candidates = [s for s in candidates if inputs.frames[job["event"]][s["effective_frame_ids"][-1]]["manual_box"]]
    selected = []
    for row in candidates:
        offset = row.get("offset_seconds") if job["stage"] == "oracle" else row["summary"]["offset_seconds"]
        if (offset_min is None or offset >= offset_min) and (offset_max is None or offset <= offset_max):
            selected.append(row)
    return selected[:limit] if limit is not None else selected


def make_jobs(inputs, stage="all", event="all", setting="default"):
    jobs = []
    for name in STAGES if stage == "all" else (stage,):
        allowed_events = ("alison", "cranston") if name in ("crop", "oracle") else ("west",)
        for key in allowed_events:
            if event not in ("all", key):
                continue
            settings = ["manual"] if name == "oracle" else list(SETTINGS) if setting == "all" else \
                inputs.protocol["default_settings"][name] if setting == "default" else [setting]
            for source_setting in settings:
                jobs.append({"stage": name, "event": key, "setting": source_setting,
                             "relative_output": f"{name}/{key}/{source_setting}"})
    require(bool(jobs), "No jobs: crop/oracle use Alison/Cranston; association/calibration use West")
    return jobs


class Results:
    """Each job writes only run.json, records.jsonl, summary.csv, summary.json."""
    def __init__(self, directory, metadata):
        self.directory, self.metadata, self.rows = directory, metadata, []
        directory.mkdir(parents=True, exist_ok=False)
        self.stream = (directory / "records.jsonl").open("x", encoding="utf-8")
        self.started = utc_now()
        self.update("running")

    def update(self, status, **extra):
        save_json(self.directory / "run.json", {**self.metadata, "status": status,
                  "started_at_utc": self.started, **extra})

    def add(self, row):
        self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()
        self.rows.append(row)

    def finish(self, error=None, diagnostics=None):
        self.stream.close()
        if self.rows:
            fields = list(dict.fromkeys(k for row in self.rows for k, v in row.items() if not isinstance(v, dict)))
            with (self.directory / "summary.csv").open("x", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                for row in self.rows:
                    writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
                                     for k, v in row.items() if k in fields})
        summary = {"records": len(self.rows), "status_counts": dict(Counter(r.get("status", "ok") for r in self.rows)),
                   "variants": dict(Counter(r.get("variant", "unknown") for r in self.rows)),
                   "interpretation": "Controlled diagnostics. Manual geometry/overlap and calibration counterfactuals are not deployment performance."}
        if diagnostics is not None:
            summary["diagnostics"] = diagnostics
        save_json(self.directory / "summary.json", summary)
        self.update("failed" if error else "completed", completed_at_utc=utc_now(), summary=summary, error=error)
