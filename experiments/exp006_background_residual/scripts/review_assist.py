"""Review-only grouping and conservative feature-based label suggestions.

Suggestions are never ground truth. Only a user's explicit acceptance writes labels.
The proposal index uses reviewed train examples, never validation/test identities.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

from common import DATA, EXPERIMENT, digest, load_manifest, now, read_json, read_jsonl, write_json


def annotation(row, state):
    """Return (reviewed, label, target). Explicit uncertainty takes precedence."""
    explicit = state.get("candidates", {}).get(row["id"])
    if explicit is not None and "label" in explicit:
        label = explicit["label"]
        return True, label, bool(explicit.get("matches_target")) if label == 1 else False
    status = state.get("frames", {}).get(Path(row["frame_file"]).stem, {}).get("status")
    if status == "uncertain":
        return True, None, False
    if status == "absent":
        return True, 0, False
    if status != "visible" and row.get("source") == "hpwren" and state.get("normal_status") == "confirmed_normal":
        return True, 0, False
    return False, None, False


def box_iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = w * h
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / area if area > 0 else 0


def group_candidates(rows, state, proposals=None, chunk_size=12, maximum_gap=120):
    """Associate overlapping crops only for the review UI, not experiment tracks.

    One-to-one matching prevents two objects at the same time sharing a group.
    Window-local tube numbers are deliberately not treated as persistent IDs.
    """
    proposals = proposals or {}
    windows = defaultdict(list)
    for row in rows:
        windows[row["timestamp"]].append(row)
    tracks = []
    for timestamp, window in sorted(windows.items()):
        edges = []
        for index, track in enumerate(tracks):
            previous = track[-1]
            if not 0 < timestamp - previous["timestamp"] <= maximum_gap:
                continue
            for item, row in enumerate(window):
                overlap = box_iou(previous["crop_box"], row["crop_box"])
                if overlap >= 0.5:
                    edges.append((overlap, index, item))
        used_tracks, used_rows = set(), set()
        for _, index, item in sorted(edges, reverse=True):
            if index not in used_tracks and item not in used_rows:
                tracks[index].append(window[item])
                used_tracks.add(index)
                used_rows.add(item)
        tracks.extend([[row] for index, row in enumerate(window) if index not in used_rows])
    groups = []
    for track_id, track in enumerate(tracks, 1):
        for start in range(0, len(track), chunk_size):
            chunk = track[start:start + chunk_size]
            pending = [r for r in chunk if not annotation(r, state)[0]]
            proposed = [proposals.get(r["id"], {}).get("label") for r in pending]
            common = proposed[0] if proposed and len(set(proposed)) == 1 else None
            ids = [r["id"] for r in chunk]
            groups.append({"id": hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16],
                "track": track_id, "part": start // chunk_size + 1, "ids": ids,
                "pending": len(pending), "reviewed": len(chunk) - len(pending),
                "suggested_label": common, "suggestion_mixed": len(set(proposed)) > 1,
                "start": chunk[0]["timestamp"], "end": chunk[-1]["timestamp"]})
    return sorted(groups, key=lambda g: (g["start"], g["track"]))


def labels_version(labels):
    return hashlib.sha256(json.dumps(labels, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save_labels(path, labels):
    """Flush a complete file before atomically replacing the prior labels."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="labels_", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(labels, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()  # Always a temporary file inside the D: data directory.


def apply_batch(state, rows, items, overwrite=False, source="human_batch"):
    """Validate the entire request before changing anything; preserve prior review."""
    if not isinstance(items, list) or not 1 <= len(items) <= 5000:
        raise ValueError("一次请选择1至5000个候选")
    by_id = {row["id"]: row for row in rows}
    seen = set()
    for item in items:
        if not isinstance(item, dict) or item.get("id") not in by_id or item["id"] in seen:
            raise ValueError("候选不存在、属于其他序列或重复；本次未保存")
        seen.add(item["id"])
        if "label" not in item or (item["label"] is not None and type(item["label"]) is not int) or item["label"] not in (0, 1, None):
            raise ValueError("候选标签必须为烟、背景或不确定")
    changed, preserved = 0, 0
    for item in items:
        row = by_id[item["id"]]
        if annotation(row, state)[0] and not overwrite:
            preserved += 1
            continue
        label = item["label"]
        state.setdefault("candidates", {})[row["id"]] = {"label": label,
            "matches_target": bool(item.get("matches_target")) if label == 1 else False,
            "source": source, "updated_at": now()}
        if row.get("source") == "hpwren" and label == 1:
            state["normal_status"] = "contains_smoke"
        changed += 1
    return {"changed": changed, "preserved": preserved}


def load_candidates(folder):
    meta_path = Path(folder) / "run.json"
    if not meta_path.exists():
        return {}
    result = {}
    for sequence in read_json(meta_path)["sequence_ids"]:
        path = Path(folder) / f"{sequence}.jsonl"
        result[sequence] = list(read_jsonl(path)) if path.exists() else []
    return result


def proposal_fingerprint(labels, sequences):
    # No val/test identities enter the cross-event appearance reference bank.
    state = {seq["id"]: labels["sequences"].get(seq["id"], {})
             for seq in sequences if seq["split"] == "train"}
    return labels_version(state)


def make_proposals(folder, sequences, labels, candidates, splits=("train", "val")):
    """DINO cosine retrieval from human-reviewed training examples, no fit/YOLO scores.

    This is an annotation aid, not an independent smoke detector or an evaluation.
    High-similarity, class-separated neighbours propose a label; others stay unknown.
    """
    import numpy as np

    folder = Path(folder)
    seq_by_id = {s["id"]: s for s in sequences}
    vectors, errors = {}, []

    def vector(row):
        if row["id"] in vectors:
            return vectors[row["id"]]
        value = None
        if row.get("feature_file"):
            path = (folder / row["feature_file"]).resolve()
            if not path.is_relative_to(folder.resolve()):
                raise ValueError("Feature path escapes the experiment folder")
            try:
                with np.load(path) as bank:
                    # Current image only; no official smoke logit/threshold is used.
                    value = bank["features"][-1].astype(np.float32).mean(axis=0)
                norm = np.linalg.norm(value)
                value = value / norm if norm > 1e-8 and np.isfinite(value).all() else None
            except (OSError, ValueError, KeyError) as error:
                errors.append({"id": row["id"], "error": str(error)})
        vectors[row["id"]] = value
        return value

    seed_by_class = {0: [], 1: []}
    # Cap highly repetitive examples per sequence/class using evenly spaced samples.
    for sequence, rows in candidates.items():
        seq = seq_by_id.get(sequence)
        if not seq or seq["split"] != "train":
            continue
        state = labels["sequences"].get(sequence, {})
        if seq["source"] == "hpwren" and state.get("normal_status") != "confirmed_normal":
            continue
        eligible = defaultdict(list)
        for row in rows:
            reviewed, label, _ = annotation(row, state)
            if reviewed and label in (0, 1) and row.get("feature_file"):
                eligible[label].append(row)
        for label, part in eligible.items():
            indexes = np.linspace(0, len(part) - 1, min(40, len(part)), dtype=int)
            for index in indexes:
                row = part[int(index)]
                value = vector(row)
                if value is not None:
                    seed_by_class[label].append((row["id"], value))
    reference = {label: np.stack([v for _, v in part]) if part else None
                 for label, part in seed_by_class.items()}
    proposals = {}
    for sequence, rows in candidates.items():
        seq = seq_by_id.get(sequence)
        if not seq or seq["split"] not in splits:
            continue
        state = labels["sequences"].get(sequence, {})
        if seq["source"] == "hpwren" and state.get("normal_status") in ("excluded", "contains_smoke"):
            continue
        for row in rows:
            if annotation(row, state)[0]:
                continue
            proposal = {"label": None, "reason": "相似证据不足，请人工判断", "method": "reviewed_train_dino_retrieval"}
            value = vector(row)
            if value is not None and all(reference[label] is not None for label in (0, 1)):
                similarities = {label: reference[label] @ value for label in (0, 1)}
                best = {label: float(similarities[label].max()) for label in (0, 1)}
                label = int(best[1] > best[0])
                margin = best[label] - best[1 - label]
                if best[label] >= 0.92 and margin >= 0.04:
                    index = int(similarities[label].argmax())
                    proposal.update(label=label, reason="与已人工确认的训练样本特征相似，待你确认",
                                    reference=seed_by_class[label][index][0])
            proposals[row["id"]] = proposal
        print(f"Prelabel: {sequence}: {sum(r['id'] in proposals for r in rows)} pending candidates", flush=True)
    result = {"schema_version": 1, "created_at": now(), "purpose": "unconfirmed_annotation_assistance_only",
        "method": "frozen_dino_current_feature_cosine_retrieval", "reference_split": "train_only",
        "seed_fingerprint": proposal_fingerprint(labels, sequences), "splits": list(splits),
        "seed_counts": {str(k): len(v) for k, v in seed_by_class.items()},
        "counts": {"smoke": sum(p["label"] == 1 for p in proposals.values()),
                   "background": sum(p["label"] == 0 for p in proposals.values()),
                   "uncertain": sum(p["label"] is None for p in proposals.values())},
        "feature_errors": errors, "candidates": proposals}
    return result


def prelabel(args):
    folder = EXPERIMENT / "outputs" / args.run_id / args.setting
    candidates = load_candidates(folder)
    if not candidates:
        raise ValueError("请先完成 Extract，并使用同一 RunId / Setting")
    splits = ("train", "val") if args.review_split == "develop" else (
        ("train", "val", "test") if args.review_split == "all" else (args.review_split,))
    path = folder / "review_suggestions.json"
    result = make_proposals(folder, load_manifest()["sequences"], read_json(DATA / "labels.json"), candidates, splits)
    apply_visual_hints(folder, result)
    # Retain suggestions for other splits when updating one split.
    if path.exists():
        previous = read_json(path).get("candidates", {})
        ids = {r["id"] for rows in candidates.values() for r in rows if r["split"] in splits}
        result["candidates"] = {k: v for k, v in previous.items() if k not in ids} | result["candidates"]
    write_json(path, result)
    print(json.dumps({"saved": str(path), "counts_this_update": result["counts"],
                      "seed_counts": result["seed_counts"], "formal_labels_changed": False}, ensure_ascii=False))
    return result


def apply_visual_hints(folder, result):
    """Optional assistant-reviewed samples + within-group propagation remain proposals."""
    path = Path(folder) / "review_visual_hints.json"
    if not path.exists():
        return
    hints = read_json(path)
    if hints.get("run_json_sha256") != digest(Path(folder) / "run.json"):
        raise ValueError("视觉预标注属于另一份提取结果，请重新查看画面后生成")
    count = 0
    for identity, hint in hints.get("candidates", {}).items():
        if identity not in result["candidates"]:
            continue  # Previously reviewed candidates are never replaced.
        label = hint.get("label")
        if label is not None and (type(label) is not int or label not in (0, 1)):
            raise ValueError("Invalid visual suggestion label")
        result["candidates"][identity] = {"label": label, "reason": hint["reason"],
            "method": "assistant_visual_samples_and_group_propagation",
            "contact_sheet": hint.get("contact_sheet"), "directly_viewed_sample": bool(hint.get("directly_viewed_sample"))}
        count += 1
    result["visual_hint_count"] = count
    result["visual_hints_sha256"] = digest(path)
    result["counts"] = {"smoke": sum(p["label"] == 1 for p in result["candidates"].values()),
                        "background": sum(p["label"] == 0 for p in result["candidates"].values()),
                        "uncertain": sum(p["label"] is None for p in result["candidates"].values())}
