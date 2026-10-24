"""Pinned classifier and association adapter. Never runs the detector."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from functools import lru_cache
import math
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from support import EXPERIMENT, ROOT, norm_to_xyxy, require, sha256


def configure_cache():
    cache = ROOT / ".cache/exp005"
    for name in ("temp", "ultralytics", "torch", "huggingface", "matplotlib"):
        (cache / name).mkdir(parents=True, exist_ok=True)
    for key, name in (("TEMP", "temp"), ("TMP", "temp"), ("YOLO_CONFIG_DIR", "ultralytics"),
                      ("TORCH_HOME", "torch"), ("HF_HOME", "huggingface"), ("MPLCONFIGDIR", "matplotlib")):
        os.environ[key] = str(cache / name)
    tempfile.tempdir = str(cache / "temp")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True


def inspect_environment(device, source_run):
    configure_cache()
    import torch
    from temporal_model.core import model as core_model
    require(device != "cuda" or torch.cuda.is_available(), "CUDA unavailable: use the documented PyTorch environment or choose --device cpu")
    folder = Path(core_model.__file__).resolve().parent
    expected = source_run["runtime_core"]["python_source_sha256"]
    checked = {}
    for name in ("model.py", "pipeline.py", "inference.py", "crop.py", "tubes.py", "stabilize.py", "types.py", "logistic_calibrator.py"):
        checked[name] = sha256(folder / name)
        require(checked[name] == expected[name], f"Installed core differs from exp004: {folder / name}")
    return {"python": sys.version, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "core_module": str(folder), "core_source_sha256": checked}


def center_scale_matches(prev, curr, threshold, iou_fn, width, height, center_gate, max_ratio):
    """Keep ordinary IoU matches preferred; only add bounded motion/scale edges."""
    candidates = []
    for i, left in enumerate(prev):
        for j, right in enumerate(curr):
            iou = iou_fn(left, right)
            if iou >= threshold:
                candidates.append((iou, i, j))
                continue
            if min(left.w, left.h, right.w, right.h) <= 0 or left.class_id != right.class_id:
                continue
            ratio = max(left.w / right.w, right.w / left.w, left.h / right.h, right.h / left.h)
            size = max(math.hypot(left.w * width, left.h * height), math.hypot(right.w * width, right.h * height))
            distance = math.hypot((left.cx - right.cx) * width, (left.cy - right.cy) * height) / size
            if ratio <= max_ratio and distance <= center_gate:
                # All fallback scores stay below the original IoU gate.
                score = threshold * 0.999 * (1 - distance / center_gate) / ratio
                candidates.append((score, i, j))
    candidates.sort(key=lambda row: -row[0])
    used_left, used_right, matches = set(), set(), []
    for score, i, j in candidates:
        if i not in used_left and j not in used_right:
            matches.append((i, j, score))
            used_left.add(i)
            used_right.add(j)
    return matches


class Runtime:
    def __init__(self, model_path, device, source_run):
        self.environment = inspect_environment(device, source_run)
        import numpy as np
        import torch
        from PIL import Image
        from temporal_model.core.crop import crop_and_resize, expand_bbox, norm_bbox_to_pixel_square
        from temporal_model.core.model import BboxTubeTemporalModel
        torch.manual_seed(0)
        self.model = BboxTubeTemporalModel.from_package(Path(model_path), device=device)
        self.np, self.torch, self.Image = np, torch, Image
        self.crop_and_resize = crop_and_resize
        self.expand, self.square = expand_bbox, norm_bbox_to_pixel_square
        self.device, self.cache = device, {}
        self.fresh_forwards = 0

    @lru_cache(maxsize=20)
    def image(self, path):
        with self.Image.open(path) as image:
            return self.np.array(image.convert("RGB"))

    def box(self, window, width, height, factor):
        return list(self.square(*self.expand(*window, factor), width, height))

    def recipe(self, inputs, event, ids, entries, window, cfg, factor):
        require(cfg["classifier"] == self.model._cfg["classifier"], "Classifier configuration drift")
        require(cfg["model_input"]["normalization"] == self.model._cfg["model_input"]["normalization"], "Normalization drift")
        require(len(entries) <= cfg["classifier"]["max_frames"], "Unexpected tube truncation")
        frame = inputs.frames[event][ids[-1]]
        box = self.box(window, frame["width"], frame["height"], factor)
        slot_ids = tuple(ids[e["frame_idx"]] if e["bbox"] is not None else None for e in entries)
        require(any(slot_ids), "No valid patches")
        require(all(inputs.frames[event][fid]["offset_seconds"] <= frame["offset_seconds"] for fid in slot_ids if fid), "Future patch")
        mi = cfg["model_input"]
        key = (event, slot_ids, tuple(box), mi["patch_size"], cfg["classifier"]["max_frames"],
               tuple(mi["normalization"]["mean"]), tuple(mi["normalization"]["std"]))
        return key, box, slot_ids

    def seed(self, inputs, event, step, cfg):
        for tube in step["prediction"]["details"]["tubes"]["kept"]:
            require(tube["stabilized_window"] is not None, "Expected stabilized cropping")
            key, _, _ = self.recipe(inputs, event, step["effective_frame_ids"], tube["entries"], tube["stabilized_window"], cfg, cfg["model_input"]["context_factor"])
            self.cache[key] = (tube["logit"], "exp004_saved")

    def score(self, inputs, event, ids, entries, window, cfg, factor, force=False):
        key, box, slot_ids = self.recipe(inputs, event, ids, entries, window, cfg, factor)
        if not force and key in self.cache:
            logit, source = self.cache[key]
            return logit, box, source
        np = self.np
        mi, maximum = cfg["model_input"], cfg["classifier"]["max_frames"]
        patches = np.zeros((maximum, 3, mi["patch_size"], mi["patch_size"]), dtype=np.float32)
        mask = np.zeros(maximum, dtype=bool)
        mean = np.asarray(mi["normalization"]["mean"], dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(mi["normalization"]["std"], dtype=np.float32).reshape(3, 1, 1)
        for index, fid in enumerate(slot_ids):
            if fid is None:
                continue
            image = self.image(str(inputs.frames[event][fid]["path"]))
            patch_image = self.crop_and_resize(image, tuple(box), mi["patch_size"])
            value = patch_image.transpose(2, 0, 1).astype(np.float32) / np.float32(255)
            patches[index] = (value - mean) / std
            mask[index] = True
        result = float(self.model._score(patches[None], mask[None])[0])
        require(math.isfinite(result), "Nonfinite classifier result")
        self.fresh_forwards += 1
        self.cache[key] = (result, "exp005_cache")
        return result, box, "fresh"

    def rebuild(self, step, cfg, variant, width, height, protocol):
        from temporal_model.core import tubes as tubes_module
        from temporal_model.core.inference import build_tubes_for_inference
        from temporal_model.core.stabilize import tube_stabilized_window
        from temporal_model.core.types import Detection, FrameDetections
        detections = [FrameDetections(frame_idx=r["frame_idx"], frame_id=r["frame_id"], timestamp=None,
                       detections=[Detection(**d) for d in r["detections"]]) for r in step["raw_detections"]]
        where = {id(det): frame.frame_idx for frame in detections for det in frame.detections}
        tc = cfg["tubes"]
        threshold = 0.1 if variant == "iou010" else tc["iou_threshold"]
        original_matcher, matches = tubes_module.match_detections, []

        def matcher(prev, curr, gate):
            if variant == "center_scale":
                result = center_scale_matches(prev, curr, gate, tubes_module.compute_iou, width, height,
                                              protocol["association_center_gate"], protocol["association_max_scale_ratio"])
            else:
                result = original_matcher(prev, curr, gate)
            for left, right, score in result:
                iou = tubes_module.compute_iou(prev[left], curr[right])
                matches.append({"from_frame_idx": where[id(prev[left])], "to_frame_idx": where[id(curr[right])],
                                "iou": iou, "association_score": score,
                                "fallback_motion_edge": variant == "center_scale" and iou < gate})
            return result

        # The patch is scoped to this single-threaded build, and always restored.
        with patch.object(tubes_module, "match_detections", matcher):
            raw = tubes_module.build_tubes(detections, iou_threshold=threshold, max_misses=tc["max_misses"])
            raw_records = [asdict(tube) for tube in raw]
            matches.clear()
            kept = build_tubes_for_inference(detections, iou_threshold=threshold, max_misses=tc["max_misses"],
                min_tube_length=tc["infer_min_tube_length"], min_detected_entries=tc["min_detected_entries"],
                interpolate_gaps=tc["interpolate_gaps"], merge_iomin=tc.get("merge_iomin"),
                merge_prox_factor=tc.get("merge_prox_factor"), merge_max_gap=tc.get("merge_max_gap"))
        records = []
        for tube in kept:
            entries = [{"frame_idx": entry.frame_idx,
                        "bbox": [entry.detection.cx, entry.detection.cy, entry.detection.w, entry.detection.h] if entry.detection else None,
                        "confidence": entry.detection.confidence if entry.detection else None,
                        "is_gap": entry.is_gap} for entry in tube.entries]
            records.append({"tube_id": tube.tube_id, "start_frame": tube.start_frame, "end_frame": tube.end_frame,
                            "entries": entries, "stabilized_window": list(tube_stabilized_window(tube.entries))})
        if variant == "original_iou020":
            saved = step["prediction"]["details"]["tubes"]["kept"]
            require(len(saved) == len(records), "Original association does not reproduce exp004")
            for actual, expected in zip(records, saved, strict=True):
                for key in ("tube_id", "start_frame", "end_frame", "entries", "stabilized_window"):
                    require(actual[key] == expected[key], f"Original association drift in {key}")
        return records, raw_records, matches
