"""Prepare/check/run/summarize exp005. --help and plan use only the standard library."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True

from support import (EXPERIMENT, ROOT, SETTINGS, STAGES, Inputs, Results, default_run_id,
                     git_commit, make_jobs, read_json, require, save_json, select_steps, sha256, utc_now)
from stages import RUNNERS, diagnostics


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "check", "run", "summary"))
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--event", choices=("all", "alison", "west", "cranston"), default="all")
    parser.add_argument("--setting", choices=("default", "all", *SETTINGS), default="default")
    parser.add_argument("--model", type=Path, default=ROOT.parent / "pyronear_lab/models/v0.6.0/model.zip")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--run-id", default=default_run_id())
    parser.add_argument("--offset-min", type=int, help="Filter window end times only, in seconds relative to official onset")
    parser.add_argument("--offset-max", type=int, help="Past frames are retained even when earlier than this output filter")
    parser.add_argument("--limit", type=int, help="Limit output window ends per job, for an explicitly separate smoke run")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id):
        parser.error("run-id must be 1..64 letters/digits/underscores/hyphens and begin with a letter or digit")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.offset_min is not None and args.offset_max is not None and args.offset_min > args.offset_max:
        parser.error("offset-min must not exceed offset-max")
    return args


def source_reference(inputs, jobs):
    job = next((j for j in jobs if j["stage"] != "oracle"), jobs[0])
    return inputs.source(job["event"], "full_baseline" if job["setting"] == "manual" else job["setting"])["run"]


def planning(inputs, jobs, args):
    plans = []
    for job in jobs:
        selected = select_steps(inputs, job, args.offset_min, args.offset_max, args.limit)
        require(bool(selected), f"No selected endpoints for {job['relative_output']}; adjust the range")
        offsets = [r["offset_seconds"] if job["stage"] == "oracle" else r["summary"]["offset_seconds"] for r in selected]
        variants = 5 if job["stage"] == "crop" else 3 if job["stage"] in ("oracle", "association") else 4
        plans.append({**job, "endpoints": len(selected), "first_minute": min(offsets) / 60,
                      "last_minute": max(offsets) / 60, "variants_per_eligible_endpoint": variants,
                      "uses_detector_inference": False, "uses_classifier": job["stage"] != "calibration"})
    return plans


def summarize(directory):
    require(directory.is_dir(), f"Run not found: {directory}")
    runs = sorted(directory.glob("*/*/*/run.json"))
    require(bool(runs), "No job records found")
    for path in runs:
        run = read_json(path)
        job = run["job"]
        print(f"{job['relative_output']}: {run['status']} | records={run.get('summary', {}).get('records', 0)}")
        for variant, result in run.get("summary", {}).get("diagnostics", {}).get("by_variant", {}).items():
            print("  " + variant + " " + json.dumps(result, ensure_ascii=False))


def main():
    args = arguments()
    directory = EXPERIMENT / "outputs" / f"exp005_{args.run_id}"
    if args.action == "summary":
        summarize(directory)
        return
    inputs = Inputs()
    jobs = make_jobs(inputs, args.stage, args.event, args.setting)
    plans = planning(inputs, jobs, args)
    if args.action == "plan":
        print(json.dumps({"output": str(directory), "jobs": plans,
                          "notes": ["No weights loaded, no inference, no output directories created.",
                                    "Crop/oracle manual positions and N counterfactuals are diagnosis only.",
                                    "Output end-time filters do not truncate the available past window."]}, ensure_ascii=False, indent=2))
        return
    gpu_jobs = any(j["stage"] != "calibration" for j in jobs)
    images = inputs.verify_images(j["event"] for j in jobs)
    environment = {"python": sys.version, "device": "standard_library_cpu"}
    if gpu_jobs:
        require(args.model.is_file(), f"Model package not found: {args.model}")
        require(sha256(args.model) == inputs.protocol["model_sha256"], "Unexpected model package SHA256")
        from runtime import inspect_environment
        environment = inspect_environment(args.device, source_reference(inputs, jobs))
    if args.action == "check":
        print(json.dumps({"status": "ready", "verified_images": images, "jobs": plans,
                          "environment": environment, "weights_loaded": False,
                          "experiment_outputs_created": False}, ensure_ascii=False, indent=2))
        return
    # Refuse the entire request before doing any work if even one job already exists.
    conflicts = [str(directory / job["relative_output"]) for job in jobs if (directory / job["relative_output"]).exists()]
    require(not conflicts, "Outputs already exist; use a new RunId or select only uncreated jobs: " + ", ".join(conflicts))
    common = {"protocol_sha256": sha256(EXPERIMENT / "protocol.json"),
              "program_sha256": {p.name: sha256(p) for p in sorted((EXPERIMENT / "scripts").glob("*.py"))},
              "source_batch": inputs.protocol["source_batch"], "model_sha256": inputs.protocol["model_sha256"],
              "requested_device": args.device, "offset_min": args.offset_min, "offset_max": args.offset_max,
              "limit": args.limit, "code_commit": git_commit()}
    batch_path = directory / "batch.json"
    if directory.exists():
        require(batch_path.is_file(), "Existing run directory has no batch identity; choose a new RunId")
        require(read_json(batch_path)["common"] == common, "RunId belongs to a different protocol/code/device/range; choose a new RunId")
    runtime = None
    if gpu_jobs:
        from runtime import Runtime
        runtime = Runtime(args.model, args.device, source_reference(inputs, jobs))
        environment = runtime.environment
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=False)
        save_json(batch_path, {"created_at_utc": utc_now(), "common": common,
                              "interpretation": "Controlled exp005 diagnostics; official final evaluation is not performed."})
    for index, job in enumerate(jobs, 1):
        selected = select_steps(inputs, job, args.offset_min, args.offset_max, args.limit)
        metadata = {"job": job, "command": sys.argv, "common": common, "environment": environment,
                    "input_sha256": inputs.provenance(job), "selected_endpoints": len(selected),
                    "manual_annotation_semantics": "Official onset stays zero; West 0..2m is not discernible; later omitted boxes reference high_resolution, not smoke absence.",
                    "uses_detector_inference": False, "compute_trigger": False}
        output = Results(directory / job["relative_output"], metadata)
        start = time.perf_counter()
        fresh_before = runtime.fresh_forwards if runtime else 0
        print(f"[{index}/{len(jobs)}] {job['relative_output']}", flush=True)
        try:
            RUNNERS[job["stage"]](inputs, job, selected, runtime, output)
            details = diagnostics(job, output.rows)
            details["fresh_classifier_forwards"] = runtime.fresh_forwards - fresh_before if runtime else 0
            details["wall_seconds"] = time.perf_counter() - start
            output.finish(diagnostics=details)
        except BaseException as error:
            output.finish(error=f"{type(error).__name__}: {error}")
            raise
    print(f"Completed {len(jobs)} jobs. Results: {directory}", flush=True)


if __name__ == "__main__":
    main()
