"""Experiment 6 entry point. Downloads, annotation, frozen features, training, test."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import sys

from common import DATA, DEFAULT_MODEL, EXPERIMENT, check_split_integrity, configure_cache, digest, load_manifest, protocol, read_json


def check(args):
    import torch
    from temporal_model.core.model import BboxTubeTemporalModel
    if not args.model.is_file() or digest(args.model) != protocol()["model_sha256"]:
        raise ValueError("Expected the pinned official v0.6.0 model.zip")
    manifest = load_manifest()
    site_splits = check_split_integrity(manifest["sequences"])
    counts = {}
    for split in ("train", "val", "test"):
        part = [s for s in manifest["sequences"] if s["split"] == split]
        frames = [f for s in part for f in s["frames"]]
        counts[split] = {"fires": sum(s["source"] == "figlib" for s in part),
            "normal_clips": sum(s["source"] == "hpwren" for s in part),
            "normal_requested_hours": sum(s.get("requested_hours", 0) for s in part),
            "images": len(frames),
            "downloaded": sum("sha256" in f and (DATA / f["file"]).is_file() for f in frames),
            "sites": len({s["site"] for s in part})}
    label_path = DATA / "labels.json"
    pending = None
    if label_path.exists():
        labels = read_json(label_path)
        pending = sum(labels["sequences"][s["id"]]["normal_status"] != "confirmed_normal"
                      for s in manifest["sequences"] if s["source"] == "hpwren")
    print(json.dumps({"python": sys.version.split()[0], "torch": str(torch.__version__),
        "device": args.device, "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "official_model_sha256": digest(args.model), "data": counts,
        "pending_normal_review_clips": pending}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check", "smoke", "data", "prepare", "review", "import-via",
        "extract", "prelabel", "predictor", "fusion", "train", "calibrate", "evaluate", "tests"])
    parser.add_argument("--run-id", default="exp006_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--setting", choices=["baseline", "high_resolution"], default="baseline")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", choices=["all", "figlib", "hpwren"], default="all")
    parser.add_argument("--sequence", action="append")
    parser.add_argument("--limit-sequences", type=int, default=0)
    parser.add_argument("--limit-frames", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--review-split", choices=["develop", "train", "val", "test", "all"], default="develop")
    parser.add_argument("--via", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id):
        parser.error("Unsafe run-id")
    if args.limit_frames < 0 or args.limit_sequences < 0 or args.workers < 1:
        parser.error("Limits must be nonnegative and workers positive")
    configure_cache()
    if args.stage == "check":
        check(args)
    elif args.stage == "smoke":
        from features import smoke
        smoke(args)
    elif args.stage == "data":
        from download_data import download
        download(args.source, args.workers)
    elif args.stage in ("prepare", "review", "import-via"):
        from review import prepare, serve, import_via
        if args.stage == "prepare":
            prepare()
        elif args.stage == "review":
            serve(args)
        elif args.via:
            import_via(args.via)
        else:
            parser.error("--via is required")
    elif args.stage == "extract":
        from features import extract
        extract(args)
    elif args.stage == "prelabel":
        from review_assist import prelabel
        prelabel(args)
    elif args.stage in ("predictor", "fusion", "train"):
        from learning import train
        train(args)
    elif args.stage in ("calibrate", "evaluate"):
        from evaluation import evaluate
        evaluate(args)
    else:
        import unittest
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.discover(str(EXPERIMENT / "tests"), pattern="test_*.py"))
        if not result.wasSuccessful():
            raise SystemExit(1)


if __name__ == "__main__":
    main()
