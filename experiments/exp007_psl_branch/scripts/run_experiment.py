"""Experiment seven: isolated data, cached features, regional PSL, review, reports."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import common7 as c


def check(args):
    import torch
    import numpy
    from temporal_model.core.model import BboxTubeTemporalModel
    if not args.model.is_file() or c.digest(args.model) != c.protocol()["model_sha256"]:
        raise ValueError("Pinned official v0.6.0 model.zip missing or mismatched")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use -Device cpu only for compatibility checks")
    for path in (c.LEGACY_DATA / "manifest.json", c.LEGACY_DATA / "labels.json",
                 c.REPLACEMENTS / "download_manifest.json", c.EXPERIMENT / "replacement_selection.json"):
        if not path.is_file():
            raise ValueError(f"Required input missing: {path}")
    print(json.dumps({"python": sys.version.split()[0], "torch": str(torch.__version__),
          "numpy": numpy.__version__, "cuda": torch.cuda.is_available(),
          "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
          "model_sha256": c.digest(args.model), "data_directory": str(c.DATA),
          "evaluation_tier": c.protocol()["evaluation_tier"]}, ensure_ascii=False, indent=2), flush=True)


def status(args):
    from data7 import readiness
    counts = {}
    for final in (False, True):
        try:
            rows = readiness(args, require_test=final)
            counts["test_ready" if final else "train_ready"] = True
            counts["candidate_counts"] = dict(Counter(r["split"] for r in rows))
        except (ValueError, FileNotFoundError) as error:
            counts["test_ready" if final else "train_ready"] = False
            counts["test_missing" if final else "train_missing"] = str(error)
    print(json.dumps(counts, ensure_ascii=False, indent=2), flush=True)
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("check", "tests", "prepare", "extract", "setup", "review", "prelabel",
        "import-via", "status", "smoke", "train", "calibrate", "evaluate", "replay", "aggregate", "run"))
    p.add_argument("--run-id", default="exp007_main_20261023")
    p.add_argument("--setting", choices=("baseline",), default="baseline")
    p.add_argument("--model", type=Path, default=c.DEFAULT_MODEL)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--variants", nargs="+", choices=("C0", "P1", "P2", "P3", "P4", "P5", "C1"))
    p.add_argument("--port", type=int, default=8767)
    p.add_argument("--review-split", choices=("train", "val", "test", "develop", "all"), default="test")
    p.add_argument("--via", type=Path)
    args = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,51}", args.run_id):
        p.error("RunId must contain 1-52 safe letters/digits/underscore/hyphen")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        p.error("Seeds must be distinct nonnegative integers")
    if args.variants and len(args.variants) != len(set(args.variants)):
        p.error("Variants must be distinct")
    if not 1024 <= args.port <= 65535:
        p.error("Port must be 1024-65535")
    c.configure_cache()
    if args.stage == "check":
        return check(args)
    if args.stage == "tests":
        import unittest
        suite = unittest.defaultTestLoader.discover(str(c.EXPERIMENT / "tests"))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
        return
    from data7 import extract, prepare, readiness
    if args.stage in ("prepare", "setup"):
        prepare(args)
        if args.stage == "prepare":
            return
    if args.stage in ("extract", "setup"):
        extract(args)
        if args.stage == "extract":
            return
    if args.stage in ("prelabel", "setup"):
        from review7 import prelabel
        prelabel(args)
        if args.stage == "setup":
            status(args)
            print(f"Next: Stage Review, http://127.0.0.1:{args.port} and /eligibility", flush=True)
        return
    if args.stage == "status":
        return status(args)
    if args.stage == "review":
        from review7 import serve
        return serve(args)
    if args.stage == "import-via":
        if not args.via:
            p.error("--via required")
        from legacy7 import bind
        return bind()["review"].import_via(args.via)
    if args.stage == "run":
        # Check the complete review state before spending time on full training.
        readiness(args, require_test=True)
    from evaluation7 import aggregate, calibrate, evaluate, replay
    if args.stage == "replay":
        return replay(args)
    if args.stage == "run":
        replay(args)
    if args.stage in ("train", "smoke", "run"):
        from learning7 import train
        for seed in (args.seeds[:1] if args.stage == "smoke" else args.seeds):
            local = SimpleNamespace(**vars(args), seed=seed)
            train(local, smoke=args.stage == "smoke")
        if args.stage != "run":
            return
    if args.stage in ("calibrate", "run"):
        for seed in args.seeds:
            calibrate(SimpleNamespace(**vars(args), seed=seed))
        if args.stage != "run":
            return
    if args.stage in ("evaluate", "run"):
        for seed in args.seeds:
            evaluate(SimpleNamespace(**vars(args), seed=seed))
        if args.stage != "run":
            return
    if args.stage in ("aggregate", "run"):
        aggregate(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FloatingPointError) as error:
        print("\nEXP007 STOP: " + str(error), file=sys.stderr, flush=True)
        raise SystemExit(2)
