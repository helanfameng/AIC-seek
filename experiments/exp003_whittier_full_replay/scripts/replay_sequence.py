"""Causal sliding-window replay for the pinned Pyronear v0.6.0 package.

Only standard-library imports run for --help and --dry-run. No model inference
is performed in either mode. Real runs preserve the package's padding policy.
"""
import argparse
import copy
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import zipfile


MODEL_SHA256 = "6a328bbc933b4d92855cbcbde06a6aac165a386d08b09e9de59f7448046d455e"
FRAME_NAME = re.compile(r"^(\d{10})_([+-]\d{5})\.jpg$", re.IGNORECASE)
ROOT = Path(__file__).resolve().parents[3]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(directory, offset_min=None, offset_max=None, limit=None):
    rows = []
    for image in directory.iterdir():
        if image.suffix.lower() != ".jpg":
            continue
        match = FRAME_NAME.fullmatch(image.name)
        if match is None:
            raise ValueError(f"Expected a FIgLib timestamp/offset filename: {image.name}")
        timestamp, offset = map(int, match.groups())
        rows.append({"path": image, "filename": image.name,
                     "timestamp": timestamp, "offset_seconds": offset})
    rows.sort(key=lambda row: row["timestamp"])
    if not rows:
        raise ValueError("No JPEG inputs found.")
    if len({row["timestamp"] - row["offset_seconds"] for row in rows}) != 1:
        raise ValueError("Files refer to different appearance origins.")
    for before, after in zip(rows, rows[1:]):
        if after["timestamp"] - before["timestamp"] != 60:
            raise ValueError("This experiment requires a complete 60-second cadence.")
    for index, row in enumerate(rows):
        row["source_frame_1based"] = index + 1
    selected = [row for row in rows
                if (offset_min is None or row["offset_seconds"] >= offset_min)
                and (offset_max is None or row["offset_seconds"] <= offset_max)]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("No images match the selected range.")
    # Verify against the acquisition manifest when one exists. File digests are
    # retained even for the old ten-frame directory, which has no manifest.
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    expected = {row["filename"]: row["sha256"] for row in manifest["files"]} if manifest else None
    for row in selected:
        row["sha256"] = sha256(row["path"])
        if expected is not None and expected.get(row["filename"]) != row["sha256"]:
            raise ValueError(f"Acquisition hash mismatch: {row['filename']}")
    return selected


def windows(rows, size):
    if size < 1:
        raise ValueError("Window size must be positive.")
    for end in range(len(rows)):
        start = max(0, end - size + 1)
        yield start, end, rows[start:end + 1]


def git_output(*args):
    try:
        result = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}",
                                 "-C", str(ROOT), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    except OSError:
        return "unavailable"


def configure_cache():
    cache = ROOT / ".cache/whittier_replay"
    for folder in [cache / "temp", cache / "ultralytics", cache / "torch", cache / "huggingface", cache / "matplotlib"]:
        folder.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(cache / "temp")
    os.environ["TMP"] = str(cache / "temp")
    tempfile.tempdir = str(cache / "temp")
    os.environ["YOLO_CONFIG_DIR"] = str(cache / "ultralytics")
    os.environ["TORCH_HOME"] = str(cache / "torch")
    os.environ["HF_HOME"] = str(cache / "huggingface")
    os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
    os.environ["XDG_CACHE_HOME"] = str(cache)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True


def summarize(rows):
    def first(predicate):
        return next(({key: row[key] for key in
                      ["source_frame_1based", "filename", "offset_seconds", "max_probability"]}
                     for row in rows if predicate(row)), None)

    runs = []
    for row in rows:
        if row["is_positive"]:
            if not runs or runs[-1]["last_step"] != row["step_1based"] - 1:
                runs.append({"start_offset_seconds": row["offset_seconds"],
                             "end_offset_seconds": row["offset_seconds"],
                             "positive_windows": 0, "last_step": row["step_1based"]})
            runs[-1]["end_offset_seconds"] = row["offset_seconds"]
            runs[-1]["positive_windows"] += 1
            runs[-1]["last_step"] = row["step_1based"]
    for run in runs:
        run.pop("last_step")
    return {
        "completed_steps": len(rows),
        "first_any_candidate": first(lambda row: row["current_candidate_count"] > 0),
        "first_kept_window": first(lambda row: row["kept_tube_count"] > 0),
        "first_positive_window": first(lambda row: row["is_positive"]),
        "first_postappearance_candidate": first(lambda row: row["offset_seconds"] >= 0 and row["current_candidate_count"] > 0),
        "first_postappearance_kept_window": first(lambda row: row["offset_seconds"] >= 0 and row["kept_tube_count"] > 0),
        "first_postappearance_positive_window": first(lambda row: row["offset_seconds"] >= 0 and row["is_positive"]),
        "preappearance_positive_windows": sum(row["offset_seconds"] < 0 and row["is_positive"] for row in rows),
        "postappearance_positive_windows": sum(row["offset_seconds"] >= 0 and row["is_positive"] for row in rows),
        "positive_window_runs": runs,
        "interpretation": "All candidate/alarm times are automatic window outputs, not manually verified correct detections. Preappearance positives are not automatically false alarms. Consecutive positive windows are not independent alarm events.",
    }


def tube_features(tube, effective_meta, n_tubes, calibrator):
    span = tube["end_frame"] - tube["start_frame"] + 1
    confidences = [entry["confidence"] for entry in tube["entries"] if entry["confidence"] is not None]
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    if not math.isfinite(tube["logit"]) or not math.isfinite(mean_conf):
        raise ValueError("Non-finite model output or confidence encountered.")
    observed = {effective_meta[entry["frame_idx"]]["filename"]
                for entry in tube["entries"] if entry["bbox"] is not None and not entry["is_gap"]}
    offsets = [effective_meta[entry["frame_idx"]]["offset_seconds"]
               for entry in tube["entries"] if entry["bbox"] is not None and not entry["is_gap"]]
    values = {"logit": tube["logit"], "log_len": math.log1p(span),
              "mean_conf": mean_conf, "n_tubes": n_tubes}
    features = {"tube_id": tube["tube_id"], "logit": tube["logit"],
                "span_frames": span, "mean_conf": mean_conf, "n_tubes": n_tubes,
                "unique_observed_frames": len(observed),
                "latest_observed_offset_seconds": max(offsets) if offsets else None,
                "probability": tube["probability"]}
    if calibrator:
        terms = {name: weight * values[name] for name, weight in
                 zip(calibrator["features"], calibrator["coefficients"], strict=True)}
        score = calibrator["intercept"] + sum(terms.values())
        p = 1 / (1 + math.exp(-score)) if score >= 0 else math.exp(score) / (1 + math.exp(score))
        if tube["probability"] is not None and abs(p - tube["probability"]) > 1e-8:
            raise ValueError("Calibrator feature decomposition disagrees with the model output.")
        features.update({"linear_terms": terms, "intercept": calibrator["intercept"],
                         "linear_score": score, "recomputed_probability": p})
    return features


def make_shapes(current, kept, effective, current_id, width, height, config, crop_functions):
    expand_bbox, to_square = crop_functions

    def rect(box):
        cx, cy, w, h = box
        return [(cx - w / 2) * width, (cy - h / 2) * height, w * width, h * height]

    shapes = [{"rect": rect([d.cx, d.cy, d.w, d.h]), "color": "#ffdc55",
               "label": f"candidate {d.confidence:.3f}"} for d in current.detections]
    for tube in kept:
        entries = [entry for entry in tube["entries"]
                   if effective[entry["frame_idx"]].frame_id == current_id and entry["bbox"] is not None]
        if not entries:
            continue  # Do not draw an old-only tube's crop on the current image.
        for entry in entries:
            shapes.append({"rect": rect(entry["bbox"]), "color": "#60ffa3",
                           "label": "gap" if entry["is_gap"] else "kept"})
        window = tube.get("stabilized_window") or entries[0]["bbox"]
        x0, y0, x1, y1 = to_square(*expand_bbox(*window, config["model_input"]["context_factor"]), width, height)
        shapes.append({"rect": [x0, y0, x1 - x0, y1 - y0], "color": "#65baff", "label": "classifier crop"})
    return shapes


VIEWER = r'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Whittier逐时刻回放</title>
<style>body{font:16px system-ui;margin:24px;background:#14202b;color:#eee}main{max-width:1500px;margin:auto}p{line-height:1.65}canvas{width:100%;height:auto}input{width:min(900px,85%)}table{border-collapse:collapse}td,th{padding:7px 12px;border-bottom:1px solid #52606c;text-align:left}a{color:#9fd4ff}</style>
<main><h1>Whittier：逐时刻滑动窗口回放</h1><p>黄：检测阈值与NMS后的候选；绿：保留轨迹；蓝：分类器当前图裁剪。阳性是窗口判断，需人工确认位置。轨迹ID只在当前窗口内有效。</p>
<label>时间点 <input id="slider" type="range" min="0" value="0"></label><p id="info"></p><canvas id="image"></canvas><p id="timing"></p><div id="features"></div>
<p>相对0秒表示官方首次可见烟羽，不是真实点火时间。保留官方重复帧填充，重复帧不等于独立观测。历史轨迹仍可使窗口为阳性；没有当前图条目的历史轨迹不在当前图上画蓝框。</p><p>数据来源：<a href="https://www.hpwren.ucsd.edu/">HPWREN / UCSD</a>，FIgLib。</p></main>
<script>const views=__VIEWS__;const slider=document.getElementById('slider'),canvas=document.getElementById('image'),ctx=canvas.getContext('2d');slider.max=views.length-1;let generation=0;const fmt=x=>x==null?'—':Number(x).toFixed(4);function draw(){const token=++generation,v=views[Number(slider.value)],r=v.row;document.getElementById('info').textContent=`原序列第${r.source_frame_1based}张 | 相对烟羽 ${r.offset_seconds/60} 分钟 | 窗口 ${r.window_real_frames} 张 | 当前候选 ${r.current_candidate_count} | 保留轨迹 ${r.kept_tube_count} | 阳性 ${r.is_positive} | 概率 ${fmt(r.max_probability)} | 填充 ${r.padded_slots}`;document.getElementById('timing').textContent=`检测 ${fmt(r.detector_ms)} ms | 管线 ${fmt(r.pipeline_wall_ms)} ms | 推理总耗时 ${fmt(r.inference_wall_ms)} ms | GPU峰值分配 ${fmt(r.peak_allocated_mb)} MiB`;const table=document.createElement('table');const head=document.createElement('tr');for(const text of ['轨迹ID','原始输出L','跨度T','检测均分C','轨迹数N','独立观测','概率']){const th=document.createElement('th');th.textContent=text;head.append(th)}table.append(head);for(const f of v.features){const tr=document.createElement('tr');for(const value of [f.tube_id,fmt(f.logit),f.span_frames,fmt(f.mean_conf),f.n_tubes,f.unique_observed_frames,fmt(f.probability)]){const td=document.createElement('td');td.textContent=value;tr.append(td)}table.append(tr)}document.getElementById('features').replaceChildren(table);const image=new Image();image.onload=()=>{if(token!==generation)return;canvas.width=v.width;canvas.height=v.height;ctx.drawImage(image,0,0);ctx.lineWidth=4;ctx.font='28px Arial';for(const box of v.shapes){ctx.strokeStyle=box.color;ctx.fillStyle=box.color;ctx.strokeRect(...box.rect);ctx.fillText(box.label,box.rect[0],Math.max(32,box.rect[1]-8))}};image.src=v.image;}slider.oninput=draw;draw();</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--detection-mode", choices=["per-frame-cache", "window-batch"], default="per-frame-cache")
    parser.add_argument("--offset-min", type=int)
    parser.add_argument("--offset-max", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Verify inputs and print the replay plan, without inference or writes.")
    parser.add_argument("--expected-model-sha256", default=MODEL_SHA256)
    args = parser.parse_args()
    if not 1 <= args.window <= 20:
        parser.error("For this v0.6.0 experiment --window must be between 1 and 20.")
    if args.confidence is not None and not 0 < args.confidence <= 1:
        parser.error("--confidence must be in (0, 1].")
    if args.image_size is not None and (args.image_size < 32 or args.image_size % 32):
        parser.error("--image-size must be a positive multiple of 32.")
    if args.warmup < 0 or (args.limit is not None and args.limit < 1):
        parser.error("Warmup must be nonnegative and limit must be positive.")
    if args.offset_min is not None and args.offset_max is not None and args.offset_min > args.offset_max:
        parser.error("offset-min must not exceed offset-max.")
    source = args.sequence.resolve()
    model_path = args.model.resolve()
    if not source.is_dir() or not model_path.is_file():
        parser.error("The sequence directory and model package must exist.")
    rows = load_inputs(source, args.offset_min, args.offset_max, args.limit)
    model_hash = sha256(model_path)
    if model_hash.lower() != args.expected_model_sha256.lower():
        parser.error("Model SHA256 differs from the expected pinned package.")
    plan = {"sequence": str(source), "selected_images": len(rows),
            "first": rows[0]["filename"], "last": rows[-1]["filename"],
            "offset_seconds": [rows[0]["offset_seconds"], rows[-1]["offset_seconds"]],
            "window_max_real_frames": args.window, "detection_mode": args.detection_mode,
            "confidence_override": args.confidence, "image_size_override": args.image_size,
            "model_sha256": model_hash, "uses_future_frames": False,
            "input_hashes_checked": True}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required for a real run.")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("Output directory already exists; choose a new run name to preserve results.")
    configure_cache()
    from PIL import Image
    import torch
    from temporal_model.core.crop import expand_bbox, norm_bbox_to_pixel_square
    from temporal_model.core.inference import pad_frames_symmetrically, pad_frames_uniform
    from temporal_model.core.model import BboxTubeTemporalModel
    from temporal_model.core.stage_timer import StageTimer
    from temporal_model.core.tubes import build_tubes

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; check the environment or explicitly choose --device cpu.")
    torch.manual_seed(0)
    output.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    run = {"status": "initializing", "started_at_utc": started, "plan": plan,
           "command": sys.argv, "script_sha256": sha256(Path(__file__)),
           "code_commit": git_output("rev-parse", "HEAD"), "code_status": git_output("status", "--short"),
           "python": sys.version, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
           "device": args.device, "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
           "selected_inputs": [{k: v for k, v in row.items() if k != "path"} for row in rows],
           "timing_scope": "Detector plus predict() wall time, CUDA synchronized; excludes package loading, warmup, diagnostic postprocessing, JSON/CSV/HTML writes and communication. Stage profiling itself adds synchronization overhead.",
           "notes": ["Preserves packaged repeated-frame padding.",
                     "Only current/past images are detected; tracks and crops are rebuilt per window.",
                     "Positive windows are not manually confirmed target alarms; historical-only tubes can keep a window positive."]}

    def save_run():
        (output / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    save_run()
    try:
        model = BboxTubeTemporalModel.from_package(model_path, device=args.device)
        core_module_path = Path(sys.modules[BboxTubeTemporalModel.__module__].__file__).resolve()
        run["runtime_core"] = {"model_module": str(core_module_path),
                               "python_source_sha256": {p.name: sha256(p) for p in sorted(core_module_path.parent.glob("*.py"))}}
        cfg = model._cfg  # Diagnostic script is tied to the pinned implementation.
        run["package_config"] = copy.deepcopy(cfg)
        if args.confidence is not None:
            cfg["infer"]["confidence_threshold"] = args.confidence
        if args.image_size is not None:
            cfg["infer"]["image_size"] = args.image_size
        if args.window > cfg["classifier"]["max_frames"]:
            raise ValueError("Requested real window exceeds the package classifier limit.")
        run["effective_config"] = copy.deepcopy(cfg)
        with zipfile.ZipFile(model_path) as archive:
            calibrator = json.loads(archive.read("logistic_calibrator.json"))
        if calibrator["features"] != ["logit", "log_len", "mean_conf", "n_tubes"]:
            raise ValueError("Unexpected calibrator feature schema.")
        run["calibrator"] = calibrator
        frames = model.load_sequence([row["path"] for row in rows])
        meta_by_id = {frame.frame_id: row for frame, row in zip(frames, rows, strict=True)}
        pad = {"symmetric": pad_frames_symmetrically, "uniform": pad_frames_uniform}[cfg["infer"].get("pad_strategy", "symmetric")]
        minimum = int(cfg["infer"].get("pad_to_min_frames", 0))

        def sync():
            if args.device == "cuda":
                torch.cuda.synchronize()

        # Warm up only the first available frame; future smoke frames are never
        # used. The first classifier-bearing step may still include cold-start cost.
        for _ in range(args.warmup):
            first_effective, _ = pad(frames[:1], min_length=minimum)
            warm_detections = model.detect(frames[:1] if args.detection_mode == "per-frame-cache" else first_effective)
            model.predict(frames[:1], frame_detections={d.frame_id: d for d in warm_detections}, compute_trigger=False)
        sync()
        run.update({"status": "running", "warmup_first_frame_repeats": args.warmup})
        save_run()
        summaries, views, cache = [], [], {}
        with (output / "steps.jsonl").open("x", encoding="utf-8") as details_file, (output / "summary.csv").open("x", encoding="utf-8-sig", newline="") as csv_file:
            writer = None
            for start, end, window_rows in windows(rows, args.window):
                window_frames = frames[start:end + 1]
                effective, padded = pad(window_frames, min_length=minimum)
                effective_meta = [meta_by_id[frame.frame_id] for frame in effective]
                current_frame = window_frames[-1]
                timer = StageTimer(args.device)
                if args.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                sync()
                wall_start = time.perf_counter()
                with timer.stage("detector"):
                    if args.detection_mode == "per-frame-cache":
                        cache[current_frame.frame_id] = model.detect([current_frame])[0]
                        detections = [replace(cache[frame.frame_id], frame_idx=i) for i, frame in enumerate(effective)]
                    else:
                        detections = model.detect(effective)
                supplied = {d.frame_id: d for d in detections}
                pipeline_start = time.perf_counter()
                prediction = model.predict(window_frames, frame_detections=supplied, timer=timer, compute_trigger=False)
                sync()
                wall_end = time.perf_counter()
                if prediction.details["preprocessing"]["num_truncated"]:
                    raise AssertionError("Unexpected truncation: sliding window construction is invalid.")
                tc = cfg["tubes"]
                raw_tubes = build_tubes(detections, iou_threshold=tc["iou_threshold"], max_misses=tc["max_misses"])
                raw_records = []
                for tube in raw_tubes:
                    observations = sum(entry.detection is not None for entry in tube.entries)
                    raw_records.append({"tube": asdict(tube), "pre_merge_filter_failures":
                                        (["short_span"] if tube.end_frame - tube.start_frame + 1 < tc["infer_min_tube_length"] else [])
                                        + (["too_few_detection_entries"] if observations < tc["min_detected_entries"] else [])})
                kept = prediction.details["tubes"]["kept"]
                feature_rows = [tube_features(tube, effective_meta, len(kept), calibrator) for tube in kept]
                winner = max(feature_rows, key=lambda f: f["probability"] if f["probability"] is not None else -1, default=None)
                current = supplied[current_frame.frame_id]
                row = {"step_1based": end + 1, "source_frame_1based": rows[end]["source_frame_1based"],
                       "filename": rows[end]["filename"], "offset_seconds": rows[end]["offset_seconds"],
                       "window_first_source_frame_1based": window_rows[0]["source_frame_1based"],
                       "window_first_offset_seconds": window_rows[0]["offset_seconds"],
                       "window_real_frames": len(window_frames), "effective_slots": len(effective),
                       "padded_slots": len(padded), "current_candidate_count": len(current.detections),
                       "raw_tube_count": len(raw_tubes), "kept_tube_count": len(kept),
                       "is_positive": bool(prediction.is_positive),
                       "max_probability": winner["probability"] if winner else None,
                       "winner_tube_id": winner["tube_id"] if winner else None,
                       "winner_latest_observed_offset_seconds": winner["latest_observed_offset_seconds"] if winner else None,
                       "winner_observation_age_seconds": rows[end]["offset_seconds"] - winner["latest_observed_offset_seconds"] if winner and winner["latest_observed_offset_seconds"] is not None else None,
                       "detector_ms": timer.as_dict().get("detector", 0.0),
                       "pipeline_wall_ms": (wall_end - pipeline_start) * 1000,
                       "inference_wall_ms": (wall_end - wall_start) * 1000,
                       "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else None,
                       "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20 if args.device == "cuda" else None}
                if writer is None:
                    writer = csv.DictWriter(csv_file, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                csv_file.flush()
                record = {"summary": row, "effective_frame_ids": [f.frame_id for f in effective],
                          "effective_source_frame_indices_1based": [m["source_frame_1based"] for m in effective_meta],
                          "raw_detections": [asdict(d) for d in detections], "raw_tubes": raw_records,
                          "prediction": asdict(prediction), "tube_features": feature_rows, "stage_ms": timer.as_dict()}
                details_file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                details_file.flush()
                with Image.open(rows[end]["path"]) as image:
                    width, height = image.size
                shapes = make_shapes(current, kept, effective, current_frame.frame_id, width, height, cfg,
                                     (expand_bbox, norm_bbox_to_pixel_square))
                views.append({"row": row, "features": feature_rows, "width": width, "height": height,
                              "image": Path(os.path.relpath(rows[end]["path"], output)).as_posix(), "shapes": shapes})
                summaries.append(row)
                # Keep detection memory bounded to the current real window.
                allowed_ids = {frame.frame_id for frame in window_frames}
                cache = {key: value for key, value in cache.items() if key in allowed_ids}
                print(f"[{end + 1:02d}/{len(rows)}] offset={row['offset_seconds']:+d}s boxes={row['current_candidate_count']} kept={len(kept)} positive={row['is_positive']} p={row['max_probability']}", flush=True)
        result = summarize(summaries)
        (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        review_fields = ["source_frame_1based", "filename", "offset_seconds", "current_candidate_count", "is_positive", "max_probability", "target_smoke_visible", "other_smoke_visible", "candidate_target_correct", "positive_target_correct", "notes"]
        with (output / "manual_review.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            review = csv.DictWriter(stream, fieldnames=review_fields)
            review.writeheader()
            for row in summaries:
                review.writerow({key: row.get(key, "") for key in review_fields})
        data_json = json.dumps(views, ensure_ascii=False).replace("<", "\\u003c")
        (output / "visualization.html").write_text(VIEWER.replace("__VIEWS__", data_json), encoding="utf-8")
        run.update({"status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "result": result})
        save_run()
        print("Saved:", output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except BaseException as error:
        run.update({"status": "failed", "error": f"{type(error).__name__}: {error}",
                    "stopped_at_utc": datetime.now(timezone.utc).isoformat()})
        save_run()
        raise


if __name__ == "__main__":
    main()
