"""Plan, check or run the frozen three-event, four-setting exp004 protocol."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys

from download_events import (CONFIG, EXPERIMENT, ROOT, event_directory,
                             load_protocol, read_json, select_events, sha256,
                             verify_event)


DEFAULT_MODEL = Path(r"D:\AIC\pyronear_lab\models\v0.6.0\model.zip")
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def replay_path(config):
    path = (ROOT / config["replay_script_relative"]).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError("Shared replay script is missing or outside this project.")
    digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if digest != config["replay_sha256_lf"]:
        raise ValueError("Shared exp003 replay changed; review and version the protocol before running.")
    return path


def run_directory(run_id):
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("RunId must contain 1-64 letters, numbers, underscores or hyphens.")
    return EXPERIMENT / "outputs" / f"exp004_{run_id}"


def build_jobs(config, model, device, run_id, event_selector="all", stage="all"):
    replay = replay_path(config)
    jobs = []
    for event in select_events(config, event_selector):
        for setting in config["settings"]:
            if stage != "all" and setting["key"] != stage:
                continue
            output = run_directory(run_id) / event["key"] / setting["key"]
            command = [sys.executable, "-B", str(replay),
                       "--sequence", str(event_directory(config, event)),
                       "--model", str(model), "--output-dir", str(output),
                       "--device", device, "--window", str(config["window_max_real_frames"]),
                       "--confidence", str(setting["confidence"]), "--image-size", str(setting["image_size"]),
                       "--detection-mode", config["detection_mode"], "--warmup", str(config["warmup"]),
                       "--expected-model-sha256", config["model_sha256"]]
            jobs.append({"event": event["key"], "setting": setting["key"],
                         "images": config["images_per_event"], "output": str(output), "command": command})
    if not jobs:
        raise ValueError("No matching experimental jobs.")
    return jobs


def refuse_existing_outputs(jobs):
    existing = [job["output"] for job in jobs if Path(job["output"]).exists()]
    if existing:
        raise FileExistsError("Requested outputs already exist; no job was started. Use a new RunId "
                              "or select only unfinished cells:\n" + "\n".join(existing))


def offline_check(config, model, event_selector):
    replay = replay_path(config)
    if not model.is_file() or sha256(model) != config["model_sha256"]:
        raise ValueError("Model package is missing or differs from the frozen v0.6.0 SHA256.")
    results, previous_hashes = [], set()
    for event in select_events(config, event_selector):
        results.append(verify_event(config, event))
        hashes = {row["sha256"] for row in read_json(event_directory(config, event) / "manifest.json")["files"]}
        if previous_hashes.intersection(hashes):
            raise ValueError("An identical JPEG occurs in different selected events.")
        previous_hashes.update(hashes)
    return {"model_sha256": config["model_sha256"], "replay_script": str(replay),
            "replay_sha256_lf": config["replay_sha256_lf"], "events": results,
            "verified_images": sum(row["verified_images"] for row in results),
            "cross_event_image_hashes_disjoint": True, "inference_executed": False}


def check_runtime(config, device):
    # Reuse the replay's existing project-local cache policy before importing ML.
    replay_api = runpy.run_path(str(replay_path(config)))
    replay_api["configure_cache"]()
    import torch
    from PIL import Image
    from temporal_model.core.model import BboxTubeTemporalModel
    from temporal_model.core.stage_timer import StageTimer
    core = Path(sys.modules[BboxTubeTemporalModel.__module__].__file__).resolve()
    available = torch.cuda.is_available()
    if device == "cuda" and not available:
        raise RuntimeError("CUDA is unavailable. Repair this environment or explicitly select --device cpu.")
    return {"python": sys.executable, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_available": available, "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "model_module": str(core), "pillow": Image.__version__,
            "core_source_sha256": {path.name: sha256(path) for path in sorted(core.parent.glob("*.py"))},
            "weights_loaded": False, "inference_executed": False}


def batch_signature(config, model, runtime):
    return {"protocol_sha256": sha256(CONFIG), "model_sha256": sha256(model),
            "replay_sha256_lf": config["replay_sha256_lf"], "runtime": runtime,
            "manifest_sha256": {event["key"]: sha256(event_directory(config, event) / "manifest.json")
                                for event in config["events"]}}


def prepare_batch(config, model, device, run_id, runtime):
    folder = run_directory(run_id)
    signature = batch_signature(config, model, runtime)
    metadata = folder / "batch.json"
    if folder.exists():
        if not metadata.is_file() or read_json(metadata)["signature"] != signature:
            raise ValueError("Existing batch lacks matching protocol/runtime/data provenance. Use a new RunId.")
        return
    folder.mkdir(parents=True)
    batch = {"experiment": "exp004_figlib_multi_event", "run_id": run_id,
             "created_at_utc": datetime.now(timezone.utc).isoformat(), "signature": signature,
             "design": build_jobs(config, model, device, run_id),
             "notes": ["Twelve planned complete replays; completion is recorded separately in each run.json.",
                       "Automatic window outputs require manual verification for target correctness."]}
    with metadata.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(batch, ensure_ascii=False, indent=2) + "\n")


def print_summary(config, model, device, run_id, event_selector, stage):
    folder = run_directory(run_id)
    if not (folder / "batch.json").is_file():
        raise FileNotFoundError(f"No actual experiment batch exists: {folder}")
    fields = ["event", "setting", "status", "completed_steps", "first_candidate_s",
              "first_kept_s", "first_positive_s", "first_postappearance_candidate_s",
              "first_postappearance_positive_s", "preappearance_positive_windows", "postappearance_positive_windows"]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    time_fields = {"first_candidate_s": "first_any_candidate", "first_kept_s": "first_kept_window",
                   "first_positive_s": "first_positive_window",
                   "first_postappearance_candidate_s": "first_postappearance_candidate",
                   "first_postappearance_positive_s": "first_postappearance_positive_window"}
    for job in build_jobs(config, model, device, run_id, event_selector, stage):
        output = Path(job["output"])
        run = read_json(output / "run.json") if (output / "run.json").is_file() else {}
        result = read_json(output / "summary.json") if (output / "summary.json").is_file() else {}
        row = {"event": job["event"], "setting": job["setting"],
               "status": run.get("status", "not_started" if not output.exists() else "incomplete")}
        for field in ["completed_steps", "preappearance_positive_windows", "postappearance_positive_windows"]:
            row[field] = result.get(field, "")
        for field, source in time_fields.items():
            row[field] = (result.get(source) or {}).get("offset_seconds", "")
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "check", "run", "summary"])
    parser.add_argument("--event", choices=["all", "alison", "west", "cranston"], default="all", type=str.lower)
    parser.add_argument("--stage", choices=["all", "full_baseline", "low_threshold", "high_resolution", "combined"], default="all")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    config, model = load_protocol(), args.model.resolve()
    jobs = build_jobs(config, model, args.device, args.run_id, args.event, args.stage)
    if args.action == "summary":
        print_summary(config, model, args.device, args.run_id, args.event, args.stage)
        return
    if args.action == "run":
        refuse_existing_outputs(jobs)  # Check every requested cell before any job begins.
    verification = offline_check(config, model, args.event)
    if args.action == "plan":
        print(json.dumps({"verification": verification, "job_count": len(jobs), "jobs": jobs}, ensure_ascii=False, indent=2))
        return
    runtime = check_runtime(config, args.device)
    print(json.dumps({"verification": verification, "runtime": runtime}, ensure_ascii=False, indent=2), flush=True)
    if args.action == "check":
        print("Preflight passed. No model weights were loaded and no inference was run.")
        return
    prepare_batch(config, model, args.device, args.run_id, runtime)
    for index, job in enumerate(jobs, 1):
        print(f"\n[{index}/{len(jobs)}] {job['event']}/{job['setting']} -> {job['output']}", flush=True)
        subprocess.run(job["command"], cwd=ROOT, check=True)
        output = Path(job["output"])
        if read_json(output / "run.json")["status"] != "completed" or read_json(output / "summary.json")["completed_steps"] != 81:
            raise RuntimeError(f"Replay did not finish all 81 frames: {output}")
    print(f"Completed {len(jobs)} requested replays. Results: {run_directory(args.run_id)}")


if __name__ == "__main__":
    main()
