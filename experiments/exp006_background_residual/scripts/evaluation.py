"""Validation-only threshold fitting, then frozen independent-event evaluation."""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import statistics

from common import DATA, EXPERIMENT, append_jsonl, digest, load_manifest, now, protocol, read_json, write_json
from learning import VARIANTS, candidate_label, load_records, require_full_run, score_rows, split_fingerprint


def normal_statistics(sequences, labels, rows, scores, variant, threshold, cfg):
    by_time = defaultdict(dict)
    for row in rows:
        if row["source"] == "hpwren":
            previous = by_time[row["sequence"]].get(row["timestamp"], 0.0)
            by_time[row["sequence"]][row["timestamp"]] = max(previous, scores[row["id"]][variant])
    episodes, positive_seconds, observed_seconds, excluded_clips = 0, 0, 0, 0
    for seq in sequences:
        if seq["source"] != "hpwren":
            continue
        state = labels["sequences"][seq["id"]]
        if state["normal_status"] in ("excluded", "contains_smoke"):
            excluded_clips += 1
            continue
        if state["normal_status"] != "confirmed_normal":
            raise ValueError(f"Normal sequence is not reviewed: {seq['id']}")
        last_positive = None
        for i, frame in enumerate(seq["frames"]):
            stamp = frame["timestamp"]
            frame_state = state["frames"].get(Path(frame["file"]).stem, {}).get("status")
            if frame_state in ("visible", "uncertain"):
                continue
            next_stamp = seq["frames"][i + 1]["timestamp"] if i + 1 < len(seq["frames"]) else seq["end_utc_exclusive"]
            duration = next_stamp - stamp
            if not 0 < duration <= cfg["maximum_history_gap_seconds"]:
                continue
            observed_seconds += duration
            if by_time[seq["id"]].get(stamp, 0.0) >= threshold:
                positive_seconds += duration
                if last_positive is None or stamp - last_positive > cfg["alarm_release_minutes"] * 60:
                    episodes += 1
                last_positive = stamp
    if not observed_seconds:
        raise ValueError("No independently reviewed normal observation time")
    hours = observed_seconds / 3600
    return {"false_alarm_episodes": episodes, "observed_camera_hours": hours, "excluded_normal_clips": excluded_clips,
            "false_alarms_per_camera_hour": episodes / hours,
            "positive_time_fraction": positive_seconds / observed_seconds,
            "positive_minutes": positive_seconds / 60}


def choose_threshold(sequences, labels, rows, scores, variant, cfg):
    values = {scores[r["id"]][variant] for r in rows if r["source"] == "hpwren"}
    candidates = sorted({0.0, 1.000001, *values, *(min(1.000001, value + 1e-7) for value in values)})
    for threshold in candidates:
        stats = normal_statistics(sequences, labels, rows, scores, variant, threshold, cfg)
        if stats["false_alarms_per_camera_hour"] <= cfg["false_alarms_per_camera_hour"] + 1e-10 and \
           stats["positive_time_fraction"] <= cfg["maximum_positive_time_fraction"] + 1e-10:
            return {"threshold": threshold, "validation_normal": stats}
    raise ValueError("No feasible threshold")


def ensure_event_review(sequences, rows, labels, setting):
    for seq in sequences:
        if seq["source"] != "figlib":
            continue
        state = labels["sequences"][seq["id"]]
        if not state.get("candidate_review_complete", {}).get(setting):
            raise ValueError(f"Complete candidate identity review before evaluation: {seq['id']} ({setting})")
        onset = seq.get("official_onset")
        if not isinstance(onset, (int, float)) or not math.isfinite(onset):
            raise ValueError(f"Official onset is missing or invalid: {seq['id']}")
        # Human first-visible time is an optional secondary reference. Requiring a
        # positive/visible target here would prevent evaluating all-background or
        # zero-candidate misses, even after every candidate was reviewed.
        human = state.get("human_visible_timestamp")
        if human is not None and not any(f["timestamp"] == human for f in seq["frames"]):
            raise ValueError(f"Human first-visible time is outside the sequence: {seq['id']}")
        for row in rows:
            if row["sequence"] != seq["id"]:
                continue
            frame_state = state["frames"].get(Path(row["frame_file"]).stem, {})
            if candidate_label(row, labels) is None and row["id"] not in state["candidates"] and frame_state.get("status") != "uncertain":
                raise ValueError(f"Candidate identity is still unreviewed: {row['id']}")


def event_statistics(sequences, labels, rows, scores, variant, threshold):
    events = []
    for seq in sequences:
        if seq["source"] != "figlib":
            continue
        state = labels["sequences"][seq["id"]]
        matching, uncertain = [], 0
        windows = defaultdict(list)
        for row in rows:
            if row["sequence"] != seq["id"] or row["timestamp"] < seq["official_onset"]:
                continue
            windows[row["timestamp"]].append(row)
        for timestamp, candidates in sorted(windows.items()):
            # Match an actual single-region alarm: the highest-scoring region wins.
            row = max(candidates, key=lambda r: scores[r["id"]][variant])
            annotation = state["candidates"].get(row["id"], {})
            if scores[row["id"]][variant] >= threshold:
                if annotation.get("label") == 1 and annotation.get("matches_target"):
                    matching.append(row["timestamp"])
                elif candidate_label(row, labels) is None:
                    uncertain += 1
        first = min(matching) if matching else None
        human = state.get("human_visible_timestamp")
        events.append({"sequence": seq["id"], "detected": first is not None,
            "delay_minutes_official": (first - seq["official_onset"]) / 60 if first is not None else None,
            "delay_minutes_human_visible": (first - human) / 60 if first is not None and human is not None else None,
            "observed_post_onset_minutes": (seq["frames"][-1]["timestamp"] - seq["official_onset"]) / 60,
            "unknown_positive_winners": uncertain})
    delays = [e["delay_minutes_official"] for e in events if e["detected"]]
    total = len(events)
    return {"events": total, "detected": len(delays), "missed_within_observation": total - len(delays),
            "detection_rate": len(delays) / total if total else None,
            "recall_at_5min": sum(d <= 5 for d in delays) / total if total else None,
            "recall_at_10min": sum(d <= 10 for d in delays) / total if total else None,
            "median_delay_detected_only": statistics.median(delays) if delays else None,
            "per_event": events}


def evaluate(args):
    from common import configure_cache
    configure_cache()
    cfg = protocol()
    folder = EXPERIMENT / "outputs" / args.run_id / args.setting
    require_full_run(folder)
    labels = read_json(DATA / "labels.json")
    sequences = load_manifest()["sequences"]
    records = load_records(folder)
    output = folder / f"seed_{args.seed}"
    threshold_path = output / "thresholds.json"
    variants = ("original", *VARIANTS)
    if args.stage == "calibrate":
        if threshold_path.exists():
            raise ValueError("Thresholds are already frozen; use a new run/seed for a changed protocol")
        import torch
        expected_labels = {s: split_fingerprint(labels, sequences, s) for s in ("train", "val")}
        for variant in VARIANTS:
            state = torch.load(output / f"{variant}.pt", map_location="cpu", weights_only=True)
            if state["label_fingerprints"] != expected_labels:
                raise ValueError("Train/val labels changed since fusion training; use a new run/seed")
        val_sequences = [s for s in sequences if s["split"] == "val"]
        rows = [r for r in records if r["split"] == "val"]
        scores = score_rows(folder, args.seed, args.device, rows)
        result = {"created_at": now(), "setting": args.setting, "seed": args.seed,
                  "validation_labels_sha256": split_fingerprint(labels, sequences, "val"),
                  "train_labels_sha256": split_fingerprint(labels, sequences, "train"),
                  "protocol_sha256": digest(EXPERIMENT / "protocol.json"),
                  "checkpoint_sha256": {v: digest(output / f"{v}.pt") for v in VARIANTS},
                  "variants": {v: choose_threshold(val_sequences, labels, rows, scores, v, cfg) for v in variants}}
        write_json(threshold_path, result)
        print("Validation thresholds frozen:", threshold_path)
        return
    thresholds = read_json(threshold_path)
    for split, key in (("val", "validation_labels_sha256"), ("train", "train_labels_sha256")):
        if thresholds[key] != split_fingerprint(labels, sequences, split):
            raise ValueError(f"{split} labels changed after threshold freezing")
    if thresholds["protocol_sha256"] != digest(EXPERIMENT / "protocol.json"):
        raise ValueError("Protocol changed after threshold freezing")
    for variant in VARIANTS:
        if thresholds["checkpoint_sha256"][variant] != digest(output / f"{variant}.pt"):
            raise ValueError("Checkpoint changed after threshold freezing")
    test_sequences = [s for s in sequences if s["split"] == "test"]
    rows = [r for r in records if r["split"] == "test"]
    ensure_event_review(test_sequences, rows, labels, args.setting)
    scores = score_rows(folder, args.seed, args.device, rows)
    result = {"created_at": now(), "setting": args.setting, "seed": args.seed, "variants": {}}
    for variant in variants:
        threshold = thresholds["variants"][variant]["threshold"]
        result["variants"][variant] = {"threshold": threshold,
            "normal": normal_statistics(test_sequences, labels, rows, scores, variant, threshold, cfg),
            "fire": event_statistics(test_sequences, labels, rows, scores, variant, threshold)}
    original_threshold = 0.7207499167831384
    result["official_default_reference"] = {"threshold": original_threshold,
        "normal": normal_statistics(test_sequences, labels, rows, scores, "original", original_threshold, cfg),
        "fire": event_statistics(test_sequences, labels, rows, scores, "original", original_threshold)}
    write_json(output / "summary.json", result)
    with (output / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            append_jsonl(handle, dict(row, scores=scores[row["id"]],
                reviewed_label=candidate_label(row, labels),
                target_annotation=labels["sequences"][row["sequence"]]["candidates"].get(row["id"])))
    lines = ["# 实验6独立测试结果", "", "阈值来自验证集；延迟相对官方首次可见时间。漏检单独计数。",
             "", "| 方法 | 检出/事件 | 漏检 | 5分钟召回 | 检出事件中位延迟/min | 误报/摄像头小时 | 超阈时长占比 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, values in result["variants"].items():
        fire, normal = values["fire"], values["normal"]
        lines.append(f"| {name} | {fire['detected']}/{fire['events']} | {fire['missed_within_observation']} | "
                     f"{fire['recall_at_5min']} | {fire['median_delay_detected_only']} | "
                     f"{normal['false_alarms_per_camera_hour']:.4f} | {normal['positive_time_fraction']:.4%} |")
    lines.extend(["", "这是本轮未参与训练和调参的测试；与官方模型原始训练数据是否重叠尚未核实。",
                  "正常观察时长较短，长期低误报能力仍需更长连续测试。", ""])
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("Independent results:", output / "RESULTS.md")
