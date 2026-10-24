"""Validation-only calibration, diagnostic comparison, frozen legacy replay."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import statistics

import common7 as c
from data7 import annotation, readiness
from legacy7 import bind


def metrics(args, rows, scores, variants, thresholds):
    old = bind()["evaluation"]
    labels, cfg = c.labels(), c.protocol()
    sequences = [s for s in c.manifest()["sequences"] if s["split"] == "test"]
    old.ensure_event_review(sequences, rows, labels, args.setting)
    result = {}
    for variant in variants:
        threshold = thresholds[variant]["threshold"]
        fire = old.event_statistics(sequences, labels, rows, scores, variant, threshold)
        # Complement the top-one rule with candidate availability and any-target
        # crossing. This diagnoses ranking without changing the primary metric.
        for event in fire["per_event"]:
            seq = next(s for s in sequences if s["id"] == event["sequence"])
            event["site"] = seq["site"]
            candidates = [r for r in rows if r["sequence"] == seq["id"] and r["timestamp"] >= seq["official_onset"]]
            targets = [r for r in candidates if annotation(r, labels)[1:] == (1, True)]
            first_candidate = min((r["timestamp"] for r in targets), default=None)
            crossing = min((r["timestamp"] for r in targets if scores[r["id"]][variant] >= threshold), default=None)
            event["first_target_candidate_minutes"] = (first_candidate - seq["official_onset"]) / 60 if first_candidate is not None else None
            event["first_any_target_crossing_minutes"] = (crossing - seq["official_onset"]) / 60 if crossing is not None else None
            winners = defaultdict(list)
            for row in candidates:
                winners[row["timestamp"]].append(row)
            event["wrong_region_positive_windows"] = sum(
                scores[max(items, key=lambda r: scores[r["id"]][variant])["id"]][variant] >= threshold and
                annotation(max(items, key=lambda r: scores[r["id"]][variant]), labels)[1:] not in ((1, True), (None, False))
                for items in winners.values())
        result[variant] = {"threshold": threshold,
            "normal": old.normal_statistics(sequences, labels, rows, scores, variant, threshold, cfg), "fire": fire}
    return result


def paired(result):
    comparisons = {}
    for ref in ("R0", "C0", "P2", "original", "mean_residual"):
        if ref not in result:
            continue
        reference = {v["sequence"]: v for v in result[ref]["fire"]["per_event"]}
        for name, values in result.items():
            if name == ref:
                continue
            differences = []
            improved = delayed = unchanged = gained = lost = 0
            for current in values["fire"]["per_event"]:
                previous = reference[current["sequence"]]
                if previous["detected"] and current["detected"]:
                    change = previous["delay_minutes_official"] - current["delay_minutes_official"]
                    differences.append(change)
                    improved += change > 1e-9
                    delayed += change < -1e-9
                    unchanged += abs(change) <= 1e-9
                gained += current["detected"] and not previous["detected"]
                lost += previous["detected"] and not current["detected"]
            comparisons[f"{name}_vs_{ref}"] = {"common_detected_events": len(differences),
                "mean_advance_minutes": statistics.mean(differences) if differences else None,
                "median_advance_minutes": statistics.median(differences) if differences else None,
                "earlier": improved, "unchanged": unchanged, "later": delayed,
                "newly_detected": gained, "newly_missed": lost}
    return comparisons


def write_results(directory, result, heading):
    c.write_json(directory / "summary.json", result)
    lines = [f"# {heading}", "", "这是修订诊断集结果，原8例已用于方案分析，不能称为新的独立测试。",
             "", "| 方法 | 检出/事件 | 5分钟召回 | 10分钟召回 | 检出事件中位延迟/min | 误报/小时 | 超阈时长占比 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, item in result["variants"].items():
        fire, normal = item["fire"], item["normal"]
        median = fire["median_delay_detected_only"]
        lines.append(f"| {name} | {fire['detected']}/{fire['events']} | {fire['recall_at_5min']:.1%} | "
                     f"{fire['recall_at_10min']:.1%} | {median if median is not None else 'NA'} | "
                     f"{normal['false_alarms_per_camera_hour']:.4f} | {normal['positive_time_fraction']:.2%} |")
    lines += ["", "阈值不在测试上调整；观察不足30小时属于当前诊断数据限制，不代表长期低误报已成立。",
              "完整逐事件、配对提前量、漏检窗口和错误区域信息见summary.json；候选分数见test_predictions.jsonl。", ""]
    (directory / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    with (directory / "per_event.csv").open("w", encoding="utf-8-sig", newline="") as out:
        writer = None
        for variant, values in result["variants"].items():
            for event in values["fire"]["per_event"]:
                row = {"variant": variant, **event}
                if writer is None:
                    writer = csv.DictWriter(out, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)


def calibrate(args):
    import torch
    from learning7 import scores
    records = readiness(args)
    directory = c.folder(args) / f"seed_{args.seed}"
    path = directory / "thresholds.json"
    if path.exists():
        validate_thresholds(args, c.read_json(path))
        print("Existing frozen thresholds reused:", path, flush=True)
        return
    rows = [r for r in records if r["split"] == "val"]
    values, variants = scores(args, rows)
    annotations, cfg = c.labels(), c.protocol()
    sequences = [s for s in c.manifest()["sequences"] if s["split"] == "val"]
    old = bind()["evaluation"]
    chosen = {v: old.choose_threshold(sequences, annotations, rows, values, v, cfg) for v in variants}
    report = {"created_at": c.now(), "variants": chosen, "seed": args.seed,
              "label_fingerprints": {s: c.fingerprint(s) for s in ("train", "val")},
              "protocol_sha256": c.digest(c.EXPERIMENT / "protocol.json"),
              "cache_identity": c.object_digest(c.require_cache(args)),
              "checkpoint_sha256": {v: c.digest(directory / f"{v}.pt") for v in variants if v != "R0"}}
    c.write_json(path, report)
    print("Validation thresholds frozen:", path, flush=True)


def validate_thresholds(args, frozen):
    import torch
    directory = c.folder(args) / f"seed_{args.seed}"
    if frozen["label_fingerprints"] != {s: c.fingerprint(s) for s in ("train", "val")} or \
       frozen["protocol_sha256"] != c.digest(c.EXPERIMENT / "protocol.json") or \
       frozen["cache_identity"] != c.object_digest(c.require_cache(args)):
        raise ValueError("Labels, protocol or cache changed after calibration")
    for variant, expected in frozen["checkpoint_sha256"].items():
        path = directory / f"{variant}.pt"
        if c.digest(path) != expected:
            raise ValueError("Checkpoint changed after calibration")
        c.assert_training_identity(torch.load(path, map_location="cpu", weights_only=True), args)


def evaluate(args):
    from learning7 import scores
    records = readiness(args, require_test=True)
    directory = c.folder(args) / f"seed_{args.seed}"
    thresholds = c.read_json(directory / "thresholds.json")
    validate_thresholds(args, thresholds)
    rows = [r for r in records if r["split"] == "test"]
    identity = {"test_labels_sha256": c.fingerprint("test"), "eligibility_sha256": c.digest(c.DATA / "eligibility.json"),
                "thresholds_sha256": c.digest(directory / "thresholds.json")}
    if (directory / "summary.json").exists():
        previous = c.read_json(directory / "summary.json")
        if previous.get("evaluation_identity") != identity:
            raise ValueError("Existing test report has different labels; preserve it and use a new RunId")
        print("Existing diagnostic evaluation reused:", directory, flush=True)
        return
    values, variants = scores(args, rows)
    measurements = metrics(args, rows, values, variants, thresholds["variants"])
    result = {"created_at": c.now(), "seed": args.seed, "evaluation_tier": c.protocol()["evaluation_tier"],
              "evaluation_identity": identity, "variants": measurements, "paired": paired(measurements),
              "head_inference_timing": c.read_json(directory / "score_timing_test.json"),
              "feature_coverage": {"all_candidates": len(rows), "with_features": sum(bool(r["feature_file"]) for r in rows)}}
    write_results(directory, result, "实验七PSL区域适配诊断结果")
    annotations = c.labels()
    c.write_jsonl(directory / "test_predictions.jsonl", [dict(r, scores=values[r["id"]],
                  reviewed_label=annotation(r, annotations)[1]) for r in rows])
    print("Diagnostic results:", directory / "RESULTS.md", flush=True)


def replay(args):
    """Existing five methods: no fitting and the original seed-42 thresholds."""
    rows = [r for r in readiness(args, require_test=True) if r["split"] == "test"]
    cfg = c.protocol()
    source = c.LEGACY / "outputs" / cfg["legacy_run_id"] / args.setting
    checkpoint_dir = source / f"seed_{cfg['legacy_seed']}"
    frozen = c.read_json(checkpoint_dir / "thresholds.json")
    for variant, expected in frozen["checkpoint_sha256"].items():
        if c.digest(checkpoint_dir / f"{variant}.pt") != expected:
            raise ValueError("Original experiment-six checkpoint changed")
    directory = c.folder(args) / "legacy_replay"
    directory.mkdir(exist_ok=True)
    identity = {"test_labels_sha256": c.fingerprint("test"),
                "eligibility_sha256": c.digest(c.DATA / "eligibility.json"),
                "source_thresholds_sha256": c.digest(checkpoint_dir / "thresholds.json")}
    if (directory / "summary.json").exists():
        if c.read_json(directory / "summary.json").get("evaluation_identity") != identity:
            raise ValueError("Replay labels changed; use a new RunId to preserve old evaluation")
        print("Existing frozen replay reused", flush=True)
        return
    old = bind()["learning"]
    values = old.score_rows(source, cfg["legacy_seed"], args.device, rows)
    variants = ["original", *old.VARIANTS]
    measured = metrics(args, rows, values, variants, frozen["variants"])
    write_results(directory, {"created_at": c.now(), "evaluation_identity": identity,
                  "variants": measured, "paired": paired(measured), "frozen_exp006_replay": True},
                  "原五种方法在修订诊断集上的冻结复评")
    c.write_jsonl(directory / "test_predictions.jsonl", [dict(r, scores=values[r["id"]]) for r in rows])


def aggregate(args):
    summaries = [c.read_json(c.folder(args) / f"seed_{seed}" / "summary.json") for seed in args.seeds]
    identities = [s["evaluation_identity"] for s in summaries]
    if len({(x["test_labels_sha256"], x["eligibility_sha256"]) for x in identities}) != 1:
        raise ValueError("Seeds evaluated against different test labels")
    names = list(summaries[0]["variants"])
    if any(set(s["variants"]) != set(names) for s in summaries):
        raise ValueError("Different variant lists across seeds")
    result = {"created_at": c.now(), "seeds": args.seeds, "evaluation_tier": c.protocol()["evaluation_tier"], "variants": {}}
    lines = ["# 实验七多种子诊断汇总", "", "每个种子单独训练和在验证集选阈值；原8例已用于开发分析。",
             "", "| 方法 | 检出事件数（各种子） | 平均误报/小时 | 平均超阈占比 |",
             "|---|---|---:|---:|"]
    for name in names:
        items = [s["variants"][name] for s in summaries]
        far = [x["normal"]["false_alarms_per_camera_hour"] for x in items]
        occupation = [x["normal"]["positive_time_fraction"] for x in items]
        detected = [x["fire"]["detected"] for x in items]
        result["variants"][name] = {"detected_by_seed": detected, "far_by_seed": far,
            "mean_far": statistics.mean(far), "mean_positive_time_fraction": statistics.mean(occupation)}
        lines.append(f"| {name} | {detected} | {statistics.mean(far):.4f} | {statistics.mean(occupation):.2%} |")
    c.write_json(c.folder(args) / "aggregate.json", result)
    (c.folder(args) / "RESULTS_ALL_SEEDS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Multi-seed report:", c.folder(args) / "RESULTS_ALL_SEEDS.md", flush=True)
