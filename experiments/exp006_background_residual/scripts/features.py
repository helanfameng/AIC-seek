"""Freeze the official model; cache original scores and causal local features."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime, timezone
import math
from pathlib import Path
import sys

from common import DATA, EXPERIMENT, ROOT, append_jsonl, check_split_integrity, configure_cache, digest, load_manifest, now, protocol, read_json, write_json


def history_indices(frames, end, length, maximum_gap):
    if end + 1 < length:
        return None
    indices = list(range(end + 1 - length, end + 1))
    times = [frames[i]["timestamp"] for i in indices]
    if any(b <= a or b - a > maximum_gap for a, b in zip(times, times[1:])):
        return None
    return indices


class FrozenRuntime:
    def __init__(self, model_path, device, setting):
        configure_cache()
        import numpy as np
        from PIL import Image
        import torch
        from temporal_model.core.model import BboxTubeTemporalModel
        from temporal_model.core.crop import crop_and_resize, expand_bbox, norm_bbox_to_pixel_square
        if digest(model_path) != protocol()["model_sha256"]:
            raise ValueError("Model differs from the pinned official v0.6.0")
        self.np, self.Image, self.torch = np, Image, torch
        self.device = device
        self.model = BboxTubeTemporalModel.from_package(Path(model_path), device=device)
        self.model._cfg["infer"].update(protocol()["settings"][setting])
        for parameter in self.model._classifier.parameters():
            parameter.requires_grad_(False)
        self.model._classifier.eval()
        original_score = self.model._score
        self.model._score = lambda patches, mask: np.concatenate(
            [original_score(patches[i:i + 2], mask[i:i + 2]) for i in range(0, len(patches), 2)])
        self.network = self.model._classifier.backbone.backbone
        self.dimension = self.model._classifier.backbone.feat_dim
        self.crop, self.expand, self.square = crop_and_resize, expand_bbox, norm_bbox_to_pixel_square
        self.images = OrderedDict()

    def image(self, path):
        key = str(path)
        if key not in self.images:
            with self.Image.open(path) as source:
                self.images[key] = self.np.asarray(source.convert("RGB"))
            if len(self.images) > 24:
                self.images.popitem(last=False)
        self.images.move_to_end(key)
        return self.images[key]

    def crop_box(self, normalized_center_box, width, height):
        factor = self.model._cfg["model_input"]["context_factor"]
        return tuple(self.square(*self.expand(*normalized_center_box, factor), width, height))

    def encode(self, frames, box):
        np, torch = self.np, self.torch
        patches = []
        mi = self.model._cfg["model_input"]
        mean = np.asarray(mi["normalization"]["mean"], dtype=np.float32)[:, None, None]
        std = np.asarray(mi["normalization"]["std"], dtype=np.float32)[:, None, None]
        for frame in frames:
            image = self.image(DATA / frame["file"])
            patch = self.crop(image, box, mi["patch_size"])
            patches.append((patch.transpose(2, 0, 1).astype(np.float32) / 255 - mean) / std)
        tensor = torch.from_numpy(np.stack(patches)).to(self.device)
        arrays = []
        grid = protocol()["grid_size"]
        with torch.inference_mode():
            for chunk in tensor.split(8):
                features = self.network.forward_features(chunk)
                tokens = features[:, self.network.num_prefix_tokens:]
                side = math.isqrt(tokens.shape[1])
                if side * side != tokens.shape[1]:
                    raise ValueError("Backbone does not expose square patch tokens")
                spatial = tokens.transpose(1, 2).reshape(len(chunk), self.dimension, side, side)
                spatial = torch.nn.functional.adaptive_avg_pool2d(spatial, (grid, grid))
                arrays.append(spatial.flatten(2).transpose(1, 2).cpu().numpy())
        return np.concatenate(arrays).astype(np.float16)


def extract(args):
    import numpy as np
    from temporal_model.core.protocol import Frame
    from temporal_model.core.types import Detection, FrameDetections
    from temporal_model.core.inference import pad_frames_symmetrically
    cfg = protocol()
    manifest = load_manifest()
    check_split_integrity(manifest["sequences"])
    sequences = manifest["sequences"]
    if args.source != "all":
        sequences = [s for s in sequences if s["source"] == args.source]
    if args.sequence:
        sequences = [s for s in sequences if s["id"] in args.sequence]
    if args.limit_sequences:
        sequences = sequences[:args.limit_sequences]
    if not sequences:
        raise ValueError("No matching sequence")
    partial = bool(args.limit_frames or args.limit_sequences or args.sequence or args.source != "all")
    folder = EXPERIMENT / "outputs" / args.run_id / args.setting
    folder.mkdir(parents=True, exist_ok=True)
    meta_path = folder / "run.json"
    identity = {"model_sha256": digest(args.model), "protocol_sha256": digest(EXPERIMENT / "protocol.json"),
                "extractor_sha256": digest(Path(__file__)),
                "manifest_selection_sha256": digest(EXPERIMENT / "data_selection.json"),
                "setting": args.setting, "partial": partial, "sequence_ids": [s["id"] for s in sequences],
                "limit_frames": args.limit_frames}
    if meta_path.exists():
        old = read_json(meta_path)
        if any(old.get(k) != value for k, value in identity.items()):
            raise ValueError("Run identity changed; choose a new --run-id")
    else:
        write_json(meta_path, dict(identity, created_at=now(), complete=False))
    runtime = FrozenRuntime(args.model, args.device, args.setting)
    features_dir = folder / "features"
    features_dir.mkdir(exist_ok=True)
    for seq in sequences:
        record_path = folder / f"{seq['id']}.jsonl"
        done_path = folder / f"{seq['id']}.complete.json"
        if done_path.exists():
            print("Cached", seq["id"], flush=True)
            continue
        entries = seq["frames"][:args.limit_frames] if args.limit_frames else seq["frames"]
        for entry in entries:
            if "sha256" not in entry or not (DATA / entry["file"]).is_file():
                raise ValueError(f"Image not downloaded: {entry['file']}")
        frames = [Frame(frame_id=Path(f["file"]).stem, image_path=DATA / f["file"],
                        timestamp=datetime.fromtimestamp(f["timestamp"], timezone.utc)) for f in entries]
        detection_path = folder / f"{seq['id']}.detections.json"
        if detection_path.exists():
            saved = read_json(detection_path)
            detections = {r["frame_id"]: FrameDetections(frame_idx=r["frame_idx"], frame_id=r["frame_id"],
                timestamp=None, detections=[Detection(**d) for d in r["detections"]]) for r in saved}
        else:
            detections = {}
            for start in range(0, len(frames), 4):
                for det in runtime.model.detect(frames[start:start + 4]):
                    detections[det.frame_id] = det
            write_json(detection_path, [asdict(detections[f.frame_id]) | {"timestamp": None} for f in frames])
        if seq["source"] == "hpwren":
            normal_file = features_dir / f"normal_{seq['id']}.npz"
            if not normal_file.exists():
                regions = []
                side = cfg["normal_training_grid"]
                width, height = entries[0]["width"], entries[0]["height"]
                for y in range(side):
                    for x in range(side):
                        regions.append((int(x * width / side), int(y * height / side),
                                        int((x + 1) * width / side), int((y + 1) * height / side)))
                bank = np.stack([runtime.encode(entries, box) for box in regions], axis=1)
                np.savez_compressed(normal_file, features=bank,
                                    timestamps=np.array([f["timestamp"] for f in entries], dtype=np.int64))
        rows = 0
        with record_path.open("w", encoding="utf-8") as output:
            for endpoint, frame in enumerate(frames):
                start = max(0, endpoint - runtime.model._cfg["classifier"]["max_frames"] + 1)
                window = frames[start:endpoint + 1]
                prediction = runtime.model.predict(window, frame_detections=detections, compute_trigger=False)
                kept = prediction.details["tubes"]["kept"]
                effective, _ = pad_frames_symmetrically(window, min_length=runtime.model._cfg["infer"]["pad_to_min_frames"])
                local_by_id = {f.frame_id: start + i for i, f in enumerate(window)}
                sequence_history = history_indices(entries, endpoint, cfg["history_frames"] + cfg["residual_steps"],
                                                   cfg["maximum_history_gap_seconds"])
                for tube in kept:
                    row_id = f"{args.setting}/{seq['id']}/{frame.frame_id}/tube{tube['tube_id']}"
                    current = entries[endpoint]
                    box = runtime.crop_box(tube["stabilized_window"], current["width"], current["height"])
                    real_entries = [e for e in tube["entries"] if not e["is_gap"] and e["bbox"] is not None]
                    last_id = effective[real_entries[-1]["frame_idx"]].frame_id
                    last_index = local_by_id[last_id]
                    confidence = sum(e["confidence"] or 0 for e in tube["entries"]) / len(tube["entries"])
                    feature_name = None
                    if sequence_history is not None:
                        feature_name = f"features/{seq['id']}_{endpoint:03d}_{tube['tube_id']}.npz"
                        target = folder / feature_name
                        if not target.exists():
                            np.savez_compressed(target, features=runtime.encode([entries[i] for i in sequence_history], box))
                    normalized_box = [box[0] / current["width"], box[1] / current["height"],
                                      box[2] / current["width"], box[3] / current["height"]]
                    row = {"id": row_id, "sequence": seq["id"], "split": seq["split"], "source": seq["source"],
                           "frame_file": current["file"], "timestamp": current["timestamp"],
                           "offset_seconds": current["offset_seconds"], "crop_box": normalized_box,
                           "logit": tube["logit"], "official_probability": tube["probability"],
                           "length": tube["end_frame"] - tube["start_frame"] + 1, "mean_confidence": confidence,
                           "n_tubes": len(kept), "age_seconds": current["timestamp"] - entries[last_index]["timestamp"],
                           "feature_file": feature_name, "history_timestamps":
                               [entries[i]["timestamp"] for i in sequence_history] if sequence_history is not None else []}
                    append_jsonl(output, row)
                    rows += 1
                if endpoint % 20 == 0:
                    print(f"{seq['id']}: {endpoint + 1}/{len(frames)} frames; {rows} candidates", flush=True)
        write_json(done_path, {"complete": True, "frames": len(frames), "rows": rows, "feature_dim": runtime.dimension})
    meta = read_json(meta_path)
    meta["complete"] = True
    meta["updated_at"] = now()
    write_json(meta_path, meta)
    print("Extraction complete:", folder, "PARTIAL" if partial else "FULL", flush=True)


def smoke(args):
    """Exercise real GPU/official-feature compatibility on an old development event."""
    from temporal_model.core.protocol import Frame
    directory = ROOT / "data/figlib_exp004_three_events/20180602_Alison_sp-s-mobo-c"
    selected = sorted(p for p in directory.glob("*.jpg") if "_+" in p.name)[:8]
    if len(selected) != 8:
        raise ValueError("The existing Alison development images are required for Smoke")
    runtime = FrozenRuntime(args.model, args.device, args.setting)
    entries = [{"file": str(p)} for p in selected]
    image = runtime.image(selected[0])
    features = runtime.encode(entries, (0, 0, image.shape[1], image.shape[0]))
    frames = [Frame(p.stem, p) for p in selected]
    detections = {}
    for start in range(0, len(frames), 4):
        for det in runtime.model.detect(frames[start:start + 4]):
            detections[det.frame_id] = det
    prediction = runtime.model.predict(frames, frame_detections=detections, compute_trigger=False)
    if features.shape != (8, protocol()["grid_size"] ** 2, runtime.dimension):
        raise ValueError(f"Unexpected feature dimensions: {features.shape}")
    if not runtime.np.isfinite(features).all():
        raise ValueError("Nonfinite frozen features")
    if any(p.requires_grad for p in runtime.model._classifier.parameters()):
        raise ValueError("Official classifier was not frozen")
    result = {"kind": "compatibility_smoke_test_only", "development_event": "Alison",
              "frames": 8, "features": list(features.shape), "device": args.device,
              "official_model_frozen": True, "kept_tubes": len(prediction.details["tubes"]["kept"]),
              "trained_new_models": False}
    path = EXPERIMENT / "outputs" / args.run_id / args.setting / "smoke.json"
    write_json(path, result)
    print(result)
