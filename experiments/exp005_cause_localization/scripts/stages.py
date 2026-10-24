"""Four controlled stages; all detections come from the completed exp004 run."""
from __future__ import annotations

import statistics

from support import calibrated, norm_to_xyxy, overlaps, require, union_box, xyxy_to_norm


def observed_entries(tube, ids):
    return [e for e in tube["entries"] if e["bbox"] is not None and not e["is_gap"]]


def features(tube, ids, n_tubes):
    observed = observed_entries(tube, ids)
    confidences = [e["confidence"] for e in tube["entries"] if e["confidence"] is not None]
    return {"span": tube["end_frame"] - tube["start_frame"] + 1,
            "confidence": sum(confidences) / len(confidences) if confidences else 0,
            "n_tubes": n_tubes, "independent_observations": len({ids[e["frame_idx"]] for e in observed})}


def region_role(inputs, event, tube, ids):
    current = inputs.frames[event][ids[-1]]
    observed = observed_entries(tube, ids)
    if not observed:
        return {"roi_group": "unassigned", "current_manual_overlap": False,
                "has_current_real_observation": False, "observation_age_seconds": None}
    latest = max(observed, key=lambda e: inputs.frames[event][ids[e["frame_idx"]]]["offset_seconds"])
    row = inputs.frames[event][ids[latest["frame_idx"]]]
    box = norm_to_xyxy(latest["bbox"], row["width"], row["height"])
    human = overlaps(row["manual_box"], box) if row["manual_box"] else None
    probe = overlaps(inputs.protocol["west_background_probe_xyxy"], box) if event == "west" else None
    is_current = row["offset_seconds"] == current["offset_seconds"]
    group = "manual_target_overlap" if human and human["iou"] > 0 else \
        "background_probe" if probe and probe["iou"] > 0 else "unassigned"
    return {"roi_group": group, "current_manual_overlap": bool(is_current and human and human["iou"] > 0),
            "has_current_real_observation": is_current,
            "observation_age_seconds": current["offset_seconds"] - row["offset_seconds"],
            "latest_observed_filename": row["filename"], "latest_box_xyxy": box,
            "manual_overlap_iou": human["iou"] if human else None,
            "probe_overlap_iou": probe["iou"] if probe else None}


def base_row(job, frame, variant):
    return {"event": job["event"], "setting": job["setting"], "offset_seconds": frame["offset_seconds"],
            "offset_minutes": frame["offset_seconds"] / 60, "filename": frame["filename"], "variant": variant,
            "visibility": frame["visibility"], "status": "ok"}


def factor_name(value):
    return f"manual_extent_c{value:g}".replace(".", "p")


def crop_stage(inputs, job, steps, runtime, output):
    source = inputs.source(job["event"], job["setting"])
    cfg, calibrator = source["run"]["effective_config"], source["run"]["calibrator"]
    tolerance = inputs.protocol["baseline_logit_tolerance"]
    for index, step in enumerate(steps, 1):
        ids = step["effective_frame_ids"]
        frame = inputs.frames[job["event"]][ids[-1]]
        kept = step["prediction"]["details"]["tubes"]["kept"]
        candidates = [tube for tube in kept if region_role(inputs, job["event"], tube, ids)["current_manual_overlap"]]
        if len(candidates) != 1:
            output.add({**base_row(job, frame, "not_scored"),
                        "status": "no_kept_target_overlap" if not candidates else "ambiguous_target_overlap",
                        "overlapping_tubes": len(candidates)})
            continue
        tube = candidates[0]
        require(tube["stabilized_window"] is not None, "Expected stabilized input")
        f = features(tube, ids, len(kept))
        saved_p = calibrated(tube["logit"], f["span"], f["confidence"], f["n_tubes"], calibrator)
        require(abs(saved_p - tube["probability"]) < inputs.protocol["calibration_tolerance"], "Saved calibration mismatch")
        human_boxes = [inputs.frames[job["event"]][ids[e["frame_idx"]]]["manual_box"] for e in tube["entries"] if e["bbox"] is not None]
        known_boxes = [box for box in human_boxes if box]
        human_union = xyxy_to_norm(union_box(known_boxes), frame["width"], frame["height"])
        original = tube["stabilized_window"]
        centered = [*human_union[:2], *original[2:]]
        contexts = inputs.protocol["crop_context_factors"]
        variants = [("original", original, cfg["model_input"]["context_factor"], "none", None),
                    ("manual_center_original_extent", centered, cfg["model_input"]["context_factor"], "center_only", "original")]
        variants.extend((factor_name(c), human_union, c, "extent_only" if c == 1.5 else "context_only",
                         "manual_center_original_extent" if c == 1.5 else factor_name(1.5)) for c in contexts)
        baseline_logit = None
        point_records = []
        for name, window, context, changed, reference in variants:
            logit, box, score_source = runtime.score(inputs, job["event"], ids, tube["entries"], window, cfg, context, force=name == "original")
            if name == "original":
                require(abs(logit - tube["logit"]) <= tolerance,
                        f"Baseline classifier does not reproduce exp004 at {frame['filename']}: {logit} vs {tube['logit']}")
                baseline_logit = logit
            probability = calibrated(logit, f["span"], f["confidence"], f["n_tubes"], calibrator)
            coverages = [overlaps(b, box)["reference_coverage"] for b in known_boxes]
            pixel_scale = cfg["model_input"]["patch_size"] / max(box[2] - box[0], box[3] - box[1])
            point_records.append({**base_row(job, frame, name), "changed_component": changed, "contrast_reference": reference,
                        "diagnostic_selection": "unique_current_real_detection_overlapping_manual_extent",
                        "source_tube_id": tube["tube_id"], "frame_ids": [ids[e["frame_idx"]] for e in tube["entries"]],
                        "slot_is_gap": [e["is_gap"] for e in tube["entries"]],
                        "frozen_T": f["span"], "frozen_C": f["confidence"], "frozen_N": f["n_tubes"],
                        "independent_observations": f["independent_observations"], "context_factor": context,
                        "crop_xyxy": box, "crop_width_pixels": box[2] - box[0], "crop_height_pixels": box[3] - box[1],
                        "current_manual_box_xyxy": frame["manual_box"],
                        "manual_box_width_in_patch_unclipped": (frame["manual_box"][2] - frame["manual_box"][0]) * pixel_scale,
                        "manual_box_height_in_patch_unclipped": (frame["manual_box"][3] - frame["manual_box"][1]) * pixel_scale,
                        "manual_union_xyxy": union_box(known_boxes), "manual_support_complete": len(known_boxes) == len(human_boxes),
                        "current_manual_box_coverage": overlaps(frame["manual_box"], box)["reference_coverage"],
                        "minimum_known_manual_box_coverage": min(coverages),
                        "raw_logit": logit, "delta_logit_vs_original": logit - baseline_logit,
                        "probability_frozen_TCN": probability, "delta_probability_vs_saved": probability - tube["probability"],
                        "saved_logit": tube["logit"], "saved_probability": tube["probability"],
                        "score_source": score_source, "automatic_detection_score": False})
        by_variant = {r["variant"]: r for r in point_records}
        for row in point_records:
            reference = by_variant[row["contrast_reference"]] if row["contrast_reference"] else row
            row["delta_logit_vs_contrast"] = row["raw_logit"] - reference["raw_logit"]
            row["delta_probability_vs_contrast"] = row["probability_frozen_TCN"] - reference["probability_frozen_TCN"]
            output.add(row)
        print(f"  crop {job['event']}/{job['setting']} {index}/{len(steps)} t={frame['offset_seconds']//60:+d}m", flush=True)


def oracle_stage(inputs, job, frames, runtime, output):
    cfg = runtime.model._cfg
    for index, frame in enumerate(frames, 1):
        all_known = [r for r in inputs.frames[job["event"]].values() if r["manual_box"]
                     and frame["offset_seconds"] - (inputs.protocol["oracle_max_frames"] - 1) * 60 <= r["offset_seconds"] <= frame["offset_seconds"]]
        if len(all_known) < inputs.protocol["oracle_min_frames"]:
            output.add({**base_row(job, frame, "not_scored"), "status": "insufficient_visible_history"})
            continue
        require(all(b["offset_seconds"] - a["offset_seconds"] == 60 for a, b in zip(all_known, all_known[1:])),
                "Oracle history contains an internal annotation gap; review it before scoring")
        ids = [r["filename"][:-4] for r in all_known]
        entries = [{"frame_idx": i, "bbox": xyxy_to_norm(r["manual_box"], r["width"], r["height"]),
                    "confidence": None, "is_gap": False} for i, r in enumerate(all_known)]
        human_union = union_box([r["manual_box"] for r in all_known])
        window = xyxy_to_norm(human_union, frame["width"], frame["height"])
        records = []
        for context in inputs.protocol["crop_context_factors"]:
            logit, box, score_source = runtime.score(inputs, job["event"], ids, entries, window, cfg, context)
            pixel_scale = cfg["model_input"]["patch_size"] / max(box[2] - box[0], box[3] - box[1])
            records.append({**base_row(job, frame, factor_name(context)), "frame_ids": ids,
                            "independent_observations": len(ids), "context_factor": context, "crop_xyxy": box,
                            "manual_union_xyxy": human_union,
                            "manual_union_width_in_patch_unclipped": (human_union[2] - human_union[0]) * pixel_scale,
                            "manual_union_height_in_patch_unclipped": (human_union[3] - human_union[1]) * pixel_scale,
                            "minimum_manual_box_coverage": min(overlaps(r["manual_box"], box)["reference_coverage"] for r in all_known),
                            "raw_logit": logit, "probability": None, "score_source": score_source,
                            "automatic_detection_score": False,
                            "calibration_note": "No detector C or retained N exists for this teacher-forced cohort; report raw L only."})
        reference = next(r["raw_logit"] for r in records if r["context_factor"] == 1.5)
        for row in records:
            output.add({**row, "delta_logit_vs_manual_c1p5": row["raw_logit"] - reference})
        print(f"  oracle {job['event']} {index}/{len(frames)} t={frame['offset_seconds']//60:+d}m", flush=True)


def association_stage(inputs, job, steps, runtime, output):
    source = inputs.source(job["event"], job["setting"])
    cfg, calibrator = source["run"]["effective_config"], source["run"]["calibrator"]
    threshold = cfg["decision"]["logistic_threshold"]
    for index, step in enumerate(steps, 1):
        ids = step["effective_frame_ids"]
        frame = inputs.frames[job["event"]][ids[-1]]
        runtime.seed(inputs, job["event"], step, cfg)
        for variant in inputs.protocol["association_variants"]:
            kept, raw, matches = runtime.rebuild(step, cfg, variant, frame["width"], frame["height"], inputs.protocol)
            details = []
            for tube in kept:
                f = features(tube, ids, len(kept))
                logit, box, score_source = runtime.score(inputs, job["event"], ids, tube["entries"], tube["stabilized_window"], cfg, cfg["model_input"]["context_factor"])
                probability = calibrated(logit, f["span"], f["confidence"], f["n_tubes"], calibrator)
                details.append({**tube, **f, **region_role(inputs, job["event"], tube, ids),
                                "raw_logit": logit, "probability": probability, "crop_xyxy": box, "score_source": score_source})
            target = [t for t in details if t["current_manual_overlap"]]
            probe = [t for t in details if t["roi_group"] == "background_probe"]
            maximum = max((t["probability"] for t in details), default=None)
            if variant == "original_iou020":
                require(len(details) == step["summary"]["kept_tube_count"], "Kept count drift")
                require((maximum is not None and maximum >= threshold) == step["summary"]["is_positive"], "Original window decision drift")
                if maximum is not None:
                    require(abs(maximum - step["summary"]["max_probability"]) < inputs.protocol["calibration_tolerance"], "Original probability drift")
            output.add({**base_row(job, frame, variant), "changed_component": "association_only",
                        "effective_frame_ids": ids, "current_candidate_count": step["summary"]["current_candidate_count"],
                        "raw_tubes": raw, "association_matches": matches, "kept_tubes": details,
                        "raw_tube_count": len(raw), "kept_tube_count": len(kept),
                        "window_probability": maximum, "window_positive": maximum is not None and maximum >= threshold,
                        "target_roi_kept": len(target) if frame["manual_box"] else None,
                        "target_roi_max_independent_observations": max((t["independent_observations"] for t in target), default=None),
                        "target_roi_max_probability": max((t["probability"] for t in target), default=None),
                        "target_roi_threshold": any(t["probability"] >= threshold for t in target) if frame["manual_box"] else None,
                        "background_probe_kept": len(probe), "background_probe_positive": sum(t["probability"] >= threshold for t in probe),
                        "background_probe_status": inputs.protocol["west_background_probe_status"],
                        "independently_adjudicated_target_identity": False})
        print(f"  association {job['setting']} {index}/{len(steps)} t={frame['offset_seconds']//60:+d}m", flush=True)


def calibration_stage(inputs, job, steps, runtime, output):
    source = inputs.source(job["event"], job["setting"])
    cfg, calibrator = source["run"]["effective_config"], source["run"]["calibrator"]
    threshold = cfg["decision"]["logistic_threshold"]
    for step in steps:
        ids = step["effective_frame_ids"]
        frame = inputs.frames[job["event"]][ids[-1]]
        kept = step["prediction"]["details"]["tubes"]["kept"]
        if not kept:
            output.add({**base_row(job, frame, "not_scored"), "status": "no_kept_tubes"})
        for tube in kept:
            f = features(tube, ids, len(kept))
            original_p = calibrated(tube["logit"], f["span"], f["confidence"], f["n_tubes"], calibrator)
            require(abs(original_p - tube["probability"]) < inputs.protocol["calibration_tolerance"], "Calibration reproduction failed")
            for name, n in (("original", f["n_tubes"]), ("N_equals_1", 1), ("add_one_irrelevant_tube", f["n_tubes"] + 1),
                            ("add_two_irrelevant_tubes", f["n_tubes"] + 2)):
                probability = calibrated(tube["logit"], f["span"], f["confidence"], n, calibrator)
                output.add({**base_row(job, frame, name), "changed_component": "calibration_N_only",
                            "tube_id": tube["tube_id"], **region_role(inputs, job["event"], tube, ids),
                            "frozen_L": tube["logit"], "frozen_T": f["span"], "frozen_C": f["confidence"],
                            "independent_observations": f["independent_observations"], "source_N": f["n_tubes"], "counterfactual_N": n,
                            "probability": probability, "saved_probability": tube["probability"],
                            "delta_probability": probability - original_p, "crosses_threshold": probability >= threshold,
                            "threshold": threshold, "retrained_calibrator": False, "automatic_detection_score": False})
    print(f"  calibration {job['setting']}: {len(steps)} saved windows, no inference", flush=True)


def diagnostics(job, rows):
    good = [r for r in rows if r["status"] == "ok"]
    result = {"scope": "Descriptive diagnostics; not formal event detection or global false-alarm scores."}
    per_variant = {}
    for variant in sorted({r["variant"] for r in good}):
        group = [r for r in good if r["variant"] == variant]
        entry = {"records": len(group)}
        if job["stage"] in ("crop", "oracle"):
            entry["median_raw_logit"] = statistics.median(r["raw_logit"] for r in group)
            delta = "delta_logit_vs_original" if job["stage"] == "crop" else "delta_logit_vs_manual_c1p5"
            entry["median_paired_delta_logit"] = statistics.median(r[delta] for r in group)
            if job["stage"] == "crop":
                entry["contrast_reference"] = group[0]["contrast_reference"]
                entry["changed_component"] = group[0]["changed_component"]
                entry["median_delta_logit_vs_single_change_contrast"] = statistics.median(r["delta_logit_vs_contrast"] for r in group)
        elif job["stage"] == "association":
            kept = [r["offset_minutes"] for r in group if r["target_roi_kept"] and r["target_roi_max_independent_observations"] >= 2]
            positive = [r["offset_minutes"] for r in group if r["target_roi_threshold"]]
            entry.update({"first_manual_overlap_retained_two_real_observations_minute": min(kept) if kept else None,
                          "first_manual_overlap_threshold_minute": min(positive) if positive else None,
                          "background_probe_positive_windows": sum(r["background_probe_positive"] > 0 for r in group)})
        else:
            entry["west_plus6_manual_overlap_examples"] = [{k: r[k] for k in ("tube_id", "frozen_L", "frozen_T", "frozen_C", "source_N", "counterfactual_N", "probability", "crosses_threshold")}
                for r in group if r["offset_seconds"] == 360 and r["current_manual_overlap"]]
        per_variant[variant] = entry
    result["by_variant"] = per_variant
    return result


RUNNERS = {"crop": crop_stage, "oracle": oracle_stage, "association": association_stage, "calibration": calibration_stage}
