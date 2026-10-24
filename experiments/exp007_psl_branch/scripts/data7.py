"""Version the revised diagnostic set, inherit labels, and reuse frozen caches."""
from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path
from types import SimpleNamespace

import common7 as c
from legacy7 import bind


def prepare(args):
    cfg = c.protocol()
    original_selection = c.LEGACY / "data_selection.json"
    original_labels = c.LEGACY_DATA / "labels.json"
    original_download = c.LEGACY_DATA / "manifest.json"
    replacement_selection = c.EXPERIMENT / "replacement_selection.json"
    replacement_download = c.REPLACEMENTS / "download_manifest.json"
    identity = {str(p.relative_to(c.ROOT)).replace("\\", "/"): c.digest(p) for p in
                (original_selection, original_download, original_labels, replacement_selection, replacement_download)}
    saved = c.DATA / "manifest.json"
    if saved.exists():
        if c.read_json(saved)["source_fingerprints"] != identity:
            raise ValueError("Source data/labels changed after Prepare; preserve the existing version")
        if not (c.DATA / "labels.json").exists():
            raise ValueError("Prepared labels missing; preserve state and restore them")
        print("Prepared data reused:", c.DATA, flush=True)
        return
    # Selection predates some byte/hash enrichment. Preserve its IDs/order but
    # take actual frame provenance from the completed download manifest.
    actual = {s["id"]: s for s in c.read_json(original_download)["sequences"]}
    originals = [actual[s["id"]] for s in c.read_json(original_selection)["sequences"]]
    replacement_map = {s["replaces_in_next_main_comparison"]: s
                       for s in c.read_json(replacement_selection)["selected_sequences"]}
    downloaded = {s["id"]: s for s in c.read_json(replacement_download)["sequences"]}
    inherited = c.read_json(original_labels)
    states, sequences, changed = {}, [], []
    for original in originals:
        replacing = replacement_map.get(original["id"])
        if replacing:
            seq = copy.deepcopy(downloaded[replacing["id"]])
            seq.update(source="figlib", split="test", replacement_for=original["id"])
            image_root = c.REPLACEMENTS
            states[seq["id"]] = {"normal_status": "unreviewed", "frames": {}, "candidates": {},
                                 "candidate_review_complete": {}, "human_visible_timestamp": None}
            changed.append(seq["id"])
        else:
            seq = copy.deepcopy(original)
            image_root = c.LEGACY_DATA
            states[seq["id"]] = copy.deepcopy(inherited["sequences"][seq["id"]])
        for frame in seq["frames"]:
            frame["file"] = str((image_root / frame["file"]).resolve())
            path = Path(frame["file"])
            if not path.is_file() or path.stat().st_size != frame["bytes"]:
                raise ValueError(f"Source image missing or wrong size: {path}")
        sequences.append(seq)
    if len(changed) != 2:
        raise ValueError("Exactly two planned replacements are required")
    # Avoid importing heavy model modules just to validate independent groups.
    site_split, event_split = {}, {}
    for seq in sequences:
        for value, seen in ((seq["site"], site_split), (seq.get("event_group"), event_split)):
            if value and value in seen and seen[value] != seq["split"]:
                raise ValueError(f"Cross-split leakage: {value}")
            if value:
                seen[value] = seq["split"]
    test_events = [s for s in sequences if s["source"] == "figlib" and s["split"] == "test"]
    if len(test_events) != cfg["expected_test_events"]:
        raise ValueError("Unexpected test denominator")
    manifest = {"created_at": c.now(), "source_fingerprints": identity, "sequences": sequences,
                "new_sequence_ids": changed, "evaluation_tier": cfg["evaluation_tier"]}
    c.DATA.mkdir(parents=True, exist_ok=True)
    c.write_json(c.EXPERIMENT / "data_selection.json", manifest)
    c.write_json(saved, manifest)
    c.write_json(c.DATA / "labels.json", {"schema_version": 1, "sequences": states})
    c.write_json(c.DATA / "eligibility.json", {"schema_version": 1, "sequences": {
        s["id"]: {"status": "pending", "reason": "Review the full source sequence under the revised scope",
                   "source": "not_yet_human_confirmed"} for s in test_events}})
    print("Prepared isolated data:", dict(Counter((s["source"], s["split"]) for s in sequences)), flush=True)


def extract(args):
    cfg = c.protocol()
    m = c.manifest()
    old_folder = c.LEGACY / "outputs" / cfg["legacy_run_id"] / args.setting
    old_meta = c.read_json(old_folder / "run.json")
    if old_meta.get("partial") or not old_meta.get("complete"):
        raise ValueError("Only a complete legacy cache can be reused")
    if old_meta["model_sha256"] != c.digest(args.model) or \
       old_meta["extractor_sha256"] != c.digest(c.LEGACY / "scripts" / "features.py") or \
       old_meta["protocol_sha256"] != c.digest(c.LEGACY / "protocol.json") or \
       old_meta["manifest_selection_sha256"] != c.digest(c.LEGACY / "data_selection.json"):
        raise ValueError("Legacy cache provenance does not match frozen sources")
    for key in ("history_frames", "residual_steps", "maximum_history_gap_seconds", "grid_size"):
        if c.read_json(c.LEGACY / "protocol.json")[key] != cfg[key]:
            raise ValueError(f"Incompatible cached feature setting: {key}")
    old_settings = c.read_json(c.LEGACY / "protocol.json")["settings"][args.setting]
    if old_settings != cfg["settings"][args.setting]:
        raise ValueError("Detection settings differ from reused cache")
    target = c.folder(args)
    identity = {"model_sha256": c.digest(args.model), "protocol_sha256": c.digest(c.EXPERIMENT / "protocol.json"),
                "manifest_selection_sha256": c.digest(c.EXPERIMENT / "data_selection.json"),
                "extractor_sha256": c.digest(c.LEGACY / "scripts" / "features.py"),
                "legacy_run_sha256": c.digest(old_folder / "run.json"),
                "setting": args.setting, "partial": False,
                "sequence_ids": [s["id"] for s in m["sequences"]]}
    if (target / "run.json").exists():
        previous = c.read_json(target / "run.json")
        if any(previous.get(k) != v for k, v in identity.items()):
            raise ValueError("Cache identity changed; choose another RunId")
        if previous.get("complete"):
            print("Complete combined cache reused:", target, flush=True)
            return
    target.mkdir(parents=True, exist_ok=True)
    c.write_json(target / "run.json", dict(identity, complete=False))
    modules = bind()
    new_args = SimpleNamespace(**vars(args))
    new_args.run_id = args.run_id + "__new"
    new_args.sequence = m["new_sequence_ids"]
    new_args.limit_sequences, new_args.limit_frames, new_args.source = 0, 0, "all"
    modules["features"].extract(new_args)
    new_folder = c.EXPERIMENT / "outputs" / new_args.run_id / args.setting
    imported, freshly_extracted = 0, 0
    record_hashes = {}
    feature_hashes = {}
    for seq in m["sequences"]:
        source = new_folder if seq["id"] in m["new_sequence_ids"] else old_folder
        done = c.read_json(source / f"{seq['id']}.complete.json")
        if not done.get("complete") or done["frames"] != len(seq["frames"]):
            raise ValueError(f"Incomplete sequence cache: {seq['id']}")
        rows = list(c.read_jsonl(source / f"{seq['id']}.jsonl"))
        if done["rows"] != len(rows):
            raise ValueError("Candidate count mismatch")
        by_time = {f["timestamp"]: f for f in seq["frames"]}
        for row in rows:
            row["frame_file"] = by_time[row["timestamp"]]["file"]
            row["feature_file"] = str((source / row["feature_file"]).resolve()) if row["feature_file"] else None
            row["cache_origin"] = str(source.relative_to(c.ROOT)).replace("\\", "/")
            if row["feature_file"]:
                ts = row["history_timestamps"]
                if len(ts) != cfg["history_frames"] + cfg["residual_steps"] or ts[-1] != row["timestamp"] or \
                   any(not 0 < b-a <= cfg["maximum_history_gap_seconds"] for a, b in zip(ts, ts[1:])):
                    raise ValueError("Invalid or noncausal feature history")
                if not Path(row["feature_file"]).is_file():
                    raise ValueError("Missing feature cache")
                if row["feature_file"] not in feature_hashes:
                    feature_hashes[row["feature_file"]] = c.digest(row["feature_file"])
        path = target / f"{seq['id']}.jsonl"
        c.write_jsonl(path, rows)
        record_hashes[seq["id"]] = c.digest(path)
        if source == old_folder:
            imported += len(rows)
        else:
            freshly_extracted += len(rows)
    c.write_json(target / "feature_sources.json", feature_hashes)
    c.write_json(target / "run.json", dict(identity, complete=True, record_sha256=record_hashes,
                 feature_sources_sha256=c.digest(target / "feature_sources.json"),
                 imported_candidates=imported, new_candidates=freshly_extracted, created_at=c.now()))
    print(f"Combined cache complete: {imported} imported + {freshly_extracted} new candidates", flush=True)


def annotation(row, annotations):
    return bind()["review_assist"].annotation(row, annotations["sequences"].get(row["sequence"], {}))


def readiness(args, require_test=False):
    rows = c.load_rows(args)
    labels = c.labels()
    issues = []
    for split in ("train", "val"):
        counts = Counter(annotation(r, labels)[1] for r in rows
                         if r["split"] == split and r["feature_file"])
        if not counts[0] or not counts[1]:
            issues.append(f"{split} needs reviewed smoke and background feature candidates")
    for seq in c.manifest()["sequences"]:
        if seq["source"] == "hpwren" and seq["split"] in (("val", "test") if require_test else ("val",)):
            if labels["sequences"][seq["id"]]["normal_status"] not in ("confirmed_normal", "excluded", "contains_smoke"):
                issues.append(f"Normal clip unreviewed: {seq['id']}")
    if require_test:
        eligibility = c.read_json(c.DATA / "eligibility.json")["sequences"]
        tests = [s for s in c.manifest()["sequences"] if s["split"] == "test" and s["source"] == "figlib"]
        for seq in tests:
            if eligibility.get(seq["id"], {}).get("status") != "eligible":
                issues.append(f"Confirm event eligibility at /eligibility: {seq['id']}")
            state = labels["sequences"][seq["id"]]
            if not state.get("candidate_review_complete", {}).get(args.setting):
                issues.append(f"Complete candidate review: {seq['id']}")
            missing = sum(not annotation(r, labels)[0] for r in rows if r["sequence"] == seq["id"])
            if missing:
                issues.append(f"{seq['id']}: {missing} candidates unreviewed")
    if issues:
        raise ValueError("\n".join(issues) + "\nUse Stage Review; source labels are never auto-confirmed.")
    return rows
