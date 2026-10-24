"""Audit the shared VIA labels and saved exp004 geometry, without inference.

Only --output-dir writes files, and existing exports are never overwritten.
Empty VIA regions are interpreted using the annotator's 2026-10-21 statement.
Geometric overlap is evidence for review, not an automatic true-positive label.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path("experiments/exp004_figlib_multi_event")
BATCH = "exp004_20261020_201923"
COVERAGE_REFERENCE = "high_resolution"  # Annotator confirmed this setting on 2026-10-21.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interpret_empty(offset, first_box, last_box):
    """Do not silently assign a meaning to a new internal annotation gap."""
    if offset < 0:
        return "before_official_onset", "not_applicable", "official_timeline"
    if first_box is None:
        return "unresolved", "unresolved", "needs_human_review"
    if offset < first_box:
        return "not_discernible", "not_localizable", "annotator_statement"
    if offset > last_box:
        return "visible", "omitted_model_covered", "annotator_statement"
    return "unresolved", "unresolved", "needs_human_review"


def xyxy_from_detection(detection, width, height):
    cx, cy, w, h = [detection[k] for k in ("cx", "cy", "w", "h")]
    return [(cx - w / 2) * width, (cy - h / 2) * height,
            (cx + w / 2) * width, (cy + h / 2) * height]


def overlap(reference, predicted):
    area_ref = (reference[2] - reference[0]) * (reference[3] - reference[1])
    area_pred = (predicted[2] - predicted[0]) * (predicted[3] - predicted[1])
    require(area_ref > 0 and area_pred > 0, "Nonpositive box area")
    intersection = max(0, min(reference[2], predicted[2]) - max(reference[0], predicted[0])) * max(
        0, min(reference[3], predicted[3]) - max(reference[1], predicted[1]))
    cx, cy = (predicted[0] + predicted[2]) / 2, (predicted[1] + predicted[3]) / 2
    return {"iou": intersection / (area_ref + area_pred - intersection),
            "reference_box_coverage": intersection / area_ref,
            "prediction_inside_reference_fraction": intersection / area_pred,
            "prediction_center_inside_reference": reference[0] <= cx <= reference[2]
            and reference[1] <= cy <= reference[3]}


def crop_box(window, width, height, context_factor):
    """Same expand -> pixel-square -> clamp order as core/crop.py."""
    cx, cy, w, h = window
    side = max(w * context_factor * width, h * context_factor * height)
    return [max(0, round(cx * width - side / 2)), max(0, round(cy * height - side / 2)),
            min(width, round(cx * width + side / 2)), min(height, round(cy * height + side / 2))]


def load_labels(root, coverage_reference):
    protocol_path = root / EXPERIMENT / "events.json"
    protocol = read_json(protocol_path)
    labels, summaries, sources = [], [], []
    for event in protocol["events"]:
        folder = root / protocol["dataset_root_relative"] / event["event_id"]
        via_files = sorted(folder.glob("via*.json"))
        require(len(via_files) == 1, f"Choose exactly one VIA project for {event['key']}")
        via_path = via_files[0]
        via = read_json(via_path)
        manifest_path = folder / "manifest.json"
        manifest = read_json(manifest_path)
        files = sorted(manifest["files"], key=lambda f: f["offset_seconds"])
        expected_offsets = list(range(protocol["offset_min_seconds"],
                                      protocol["offset_max_seconds"] + 1,
                                      protocol["interval_seconds"]))
        require([f["offset_seconds"] for f in files] == expected_offsets, "Manifest timeline changed")
        require(len(files) == protocol["images_per_event"], "Unexpected event length")
        metadata = via["_via_img_metadata"]
        require(len(via["_via_image_id_list"]) == len(set(via["_via_image_id_list"]))
                == len(metadata), "Duplicate or missing VIA IDs")
        require(set(via["_via_image_id_list"]) == set(metadata), "VIA ID list disagrees with metadata")
        by_name = {v["filename"]: v for v in metadata.values()}
        require(len(by_name) == len(metadata), "Duplicate VIA filenames")
        require(set(by_name) == {f["filename"] for f in files}, "VIA/manifest filename mismatch")
        box_offsets = [f["offset_seconds"] for f in files if by_name[f["filename"]]["regions"]]
        first_box, last_box = (min(box_offsets), max(box_offsets)) if box_offsets else (None, None)
        event_labels = []
        for index, item in enumerate(files, 1):
            filename, offset = item["filename"], item["offset_seconds"]
            image_path = folder / filename
            require(filename == f"{event['origin_timestamp'] + offset}_{offset:+06d}.jpg",
                    f"Filename/onset disagreement: {filename}")
            record = by_name[filename]
            require(image_path.stat().st_size == item["bytes"] == record["size"],
                    f"Image byte size mismatch: {filename}")
            require(digest(image_path) == item["sha256"], f"Image hash mismatch: {filename}")
            boxes = []
            for region in record["regions"]:
                shape = region["shape_attributes"]
                require(shape["name"] == "rect", "Only rectangular smoke extents are supported")
                x, y, w, h = [shape[k] for k in ("x", "y", "width", "height")]
                require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in (x, y, w, h)),
                        "Invalid annotation coordinates")
                require(w > 0 and h > 0 and 0 <= x < x + w <= item["width"]
                        and 0 <= y < y + h <= item["height"], f"Out-of-bounds annotation: {filename}")
                require(offset >= 0, "Pre-onset box requires human review of the onset/region identity")
                boxes.append([x, y, x + w, y + h])
            if boxes:
                visibility, localization, basis = "visible", "manual_coarse_box", "via_region"
            else:
                visibility, localization, basis = interpret_empty(offset, first_box, last_box)
            row = {"image_id": f"{event['event_id']}/{filename}", "event": event["key"],
                   "event_id": event["event_id"], "filename": filename, "source_frame_1based": index,
                   "offset_seconds": offset, "image_sha256": item["sha256"],
                   "width": item["width"], "height": item["height"], "target_visibility": visibility,
                   "localization_status": localization, "label_basis": basis,
                   "manual_boxes_xyxy": boxes, "coordinate_space": "original_image_pixels",
                   "geometry_evaluable": bool(boxes),
                   "coverage_reference_setting": coverage_reference if localization == "omitted_model_covered" else None,
                   "coverage_verified_for_all_four_settings": False,
                   "other_smoke_status": "not_recorded", "background_regions": [],
                   "background_review_status": "not_exhaustively_reviewed",
                   "uncertainty_review_status": "not_recorded"}
            event_labels.append(row)
        labels.extend(event_labels)
        summaries.append({"event": event["key"], "frames": len(event_labels),
                          "manual_box_frames": len(box_offsets),
                          "manual_box_count": sum(len(r["manual_boxes_xyxy"]) for r in event_labels),
                          "manual_box_minutes": [v // 60 for v in box_offsets],
                          "first_human_discernible_offset_seconds": first_box,
                          "last_manual_box_offset_seconds": last_box,
                          "visibility_counts": dict(Counter(r["target_visibility"] for r in event_labels)),
                          "localization_counts": dict(Counter(r["localization_status"] for r in event_labels)),
                          "nonempty_via_file_attributes": sum(bool(r.get("file_attributes")) for r in metadata.values()),
                          "nonempty_via_region_attributes": sum(bool(r.get("region_attributes"))
                              for item in metadata.values() for r in item["regions"])})
        for path in (via_path, manifest_path):
            sources.append({"path": path.relative_to(root).as_posix(), "sha256": digest(path)})
    require(len({r["image_id"] for r in labels}) == len(labels), "Duplicate image IDs")
    require(len({r["image_sha256"] for r in labels}) == len(labels), "Duplicate image content needs review")
    sources.append({"path": protocol_path.relative_to(root).as_posix(), "sha256": digest(protocol_path)})
    return protocol, labels, summaries, sources


def collect_geometry(root, protocol, labels):
    lookup = {(r["event"], r["filename"]): r for r in labels}
    evidence, runs, sources = [], [], []
    for event in protocol["events"]:
        event_labels = [r for r in labels if r["event"] == event["key"]]
        for setting in protocol["settings"]:
            folder = root / EXPERIMENT / "outputs" / BATCH / event["key"] / setting["key"]
            run_path, steps_path = folder / "run.json", folder / "steps.jsonl"
            run = read_json(run_path)
            require(run["status"] == "completed", "Incomplete model run")
            actual = run["effective_config"]
            require(actual["infer"]["image_size"] == setting["image_size"]
                    and actual["infer"]["confidence_threshold"] == setting["confidence"], "Setting mismatch")
            steps = [json.loads(line) for line in steps_path.read_text(encoding="utf-8").splitlines()]
            require(len(steps) == len(event_labels), "Missing prediction steps")
            for step, label in zip(steps, event_labels):
                summary = step["summary"]
                require(summary["filename"] == label["filename"]
                        and summary["offset_seconds"] == label["offset_seconds"], "Step/label timeline mismatch")
                ids = step["effective_frame_ids"]
                require(ids[-1] == Path(label["filename"]).stem, "Current frame is not last effective frame")
                require(all(lookup[(event["key"], frame + ".jpg")]["offset_seconds"]
                            <= label["offset_seconds"] for frame in ids), "Future frame in saved window")
                if not label["geometry_evaluable"]:
                    continue
                # Current dataset uses one coarse extent per frame, as declared by the annotator.
                require(len(label["manual_boxes_xyxy"]) == 1, "Multiple target boxes require explicit matching")
                reference = label["manual_boxes_xyxy"][0]
                width, height = label["width"], label["height"]
                detections = step["raw_detections"][-1]
                require(detections["frame_id"] == ids[-1], "Current detection/frame mismatch")
                candidates = []
                for index, detection in enumerate(detections["detections"]):
                    box = xyxy_from_detection(detection, width, height)
                    candidates.append({"candidate_index": index, "confidence": detection["confidence"],
                                       "bbox_xyxy": box, **overlap(reference, box)})
                kept_evidence = []
                for tube in step["prediction"]["details"]["tubes"]["kept"]:
                    current_entries = [entry for entry in tube["entries"] if ids[entry["frame_idx"]] == ids[-1]
                                       and entry["bbox"] is not None and not entry["is_gap"]]
                    if not current_entries:
                        continue  # No current real localization: do not label a stale/gap box as a detection.
                    require(actual["model_input"]["stabilize"] and tube["stabilized_window"] is not None,
                            "Crop reconstruction requires the saved stabilized window")
                    entry = current_entries[-1]
                    det = dict(zip(("cx", "cy", "w", "h"), entry["bbox"]))
                    det_box = xyxy_from_detection(det, width, height)
                    crop = crop_box(tube["stabilized_window"], width, height,
                                    actual["model_input"]["context_factor"])
                    historical = []
                    observed_ids = set()
                    for observation in tube["entries"]:
                        if observation["bbox"] is None or observation["is_gap"]:
                            continue
                        filename = ids[observation["frame_idx"]] + ".jpg"
                        if filename in observed_ids:
                            continue  # Padding slots are not independent real observations.
                        observed_ids.add(filename)
                        historical_label = lookup[(event["key"], filename)]
                        if historical_label["geometry_evaluable"]:
                            historical.append({"filename": filename,
                                               "crop_reference_box_coverage": overlap(
                                                   historical_label["manual_boxes_xyxy"][0], crop)["reference_box_coverage"]})
                    kept_evidence.append({"tube_id": tube["tube_id"], "current_bbox_xyxy": det_box,
                                          "current_box_overlap": overlap(reference, det_box),
                                          "crop_xyxy": crop,
                                          "crop_reference_box_coverage": overlap(reference, crop)["reference_box_coverage"],
                                          "logit": tube["logit"], "probability": tube["probability"],
                                          "independent_observed_frames": len(observed_ids),
                                          "annotated_historical_frames": historical,
                                          "is_true_target": None})
                evidence.append({"event": event["key"], "setting": setting["key"],
                                 "filename": label["filename"], "offset_seconds": label["offset_seconds"],
                                 "manual_box_xyxy": reference, "candidates": candidates,
                                 "kept_with_current_real_observation": kept_evidence})
            runs.append({"event": event["key"], "setting": setting["key"], "steps": len(steps),
                         "logistic_threshold": actual["decision"]["logistic_threshold"]})
            for path in (run_path, steps_path):
                sources.append({"path": path.relative_to(root).as_posix(), "sha256": digest(path)})
    return evidence, runs, sources


def build_audit(root=PROJECT_ROOT, coverage_reference=COVERAGE_REFERENCE):
    protocol, labels, events, sources = load_labels(Path(root), coverage_reference)
    geometry, runs, run_sources = collect_geometry(Path(root), protocol, labels)
    audit = {"schema_version": 1, "annotator_statement_date": "2026-10-21",
             "batch": BATCH, "coverage_reference_setting": coverage_reference,
             "coverage_reference_basis": "annotator_confirmation_2026-10-21"
             if coverage_reference == COVERAGE_REFERENCE else "explicit_argument",
             "scope": "Shared human annotation semantics and geometric evidence; no new inference or official accuracy scores.",
             "totals": {"unique_images": len(labels),
                        "manual_box_frames": sum(r["geometry_evaluable"] for r in labels),
                        "visibility_counts": dict(Counter(r["target_visibility"] for r in labels)),
                        "localization_counts": dict(Counter(r["localization_status"] for r in labels)),
                        "audited_prediction_steps": sum(r["steps"] for r in runs),
                        "annotated_frame_setting_pairs": len(geometry)},
             "events": events, "runs": runs, "sources": sources + run_sources,
             "limitations": ["Empty VIA regions have no intrinsic class label; phase meanings use the annotator's statement.",
                             "Late model-covered frames have no independent localization geometry.",
                             "Late coverage confirmation refers to the recorded reference setting; the other settings need their own output review.",
                             "Other smoke, exhaustive difficult-background regions and explicit uncertainty flags are not encoded.",
                             "Box overlap and reconstructed crop coverage do not establish candidate identity or causal model failure.",
                             "No actual alert/reset/notification behavior was replayed."],
             "geometry_evidence": geometry}
    return labels, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--coverage-reference", default=COVERAGE_REFERENCE,
                        choices=("full_baseline", "low_threshold", "high_resolution", "combined"))
    parser.add_argument("--output-dir", type=Path, help="Export two new files; refuse existing output files")
    args = parser.parse_args()
    labels, audit = build_audit(args.project_root, args.coverage_reference)
    if args.output_dir:
        paths = [args.output_dir / "frame_labels.jsonl", args.output_dir / "audit.json"]
        require(not any(p.exists() for p in paths), "An export already exists; use a new output directory")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with paths[0].open("x", encoding="utf-8", newline="\n") as stream:
            for row in labels:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        with paths[1].open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(audit, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"totals": audit["totals"], "events": audit["events"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
