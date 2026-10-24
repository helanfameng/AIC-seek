"""Read-only checkpoint diagnosis; writes only this script's diagnostic JSON.

Run with the existing Pyronear package directory on sys.path. No training,
threshold fitting, source-result modification, or bytecode generation occurs.
Batch geometry is evaluated at saved best checkpoints, not replayed training.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import site
import sys

sys.dont_write_bytecode = True
site.addsitedir(r"D:\AIC\pyronear_lab\.venv\Lib\site-packages")
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "scripts"))

import numpy as np
import torch
from torch.nn import functional as F

from learning7 import base_features, current_features
from models7 import RegionalHead, assign_to_prototypes, scatter_matrices, bes_loss

torch.set_num_threads(2)
RUN = EXP / "outputs" / "exp007_main_20261023" / "baseline"
DATA = EXP.parents[1] / "data" / "exp007_psl_branch"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


LABELS = read(DATA / "labels.json")
CFG = read(EXP / "protocol.json")
ROWS = []
for path in sorted(RUN.glob("*.jsonl")):
    ROWS.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def annotation(row):
    state = LABELS["sequences"][row["sequence"]]
    entry = state.get("candidates", {}).get(row["id"])
    if entry is not None and "label" in entry:
        return entry["label"], bool(entry.get("matches_target")) if entry["label"] == 1 else False
    frame = state.get("frames", {}).get(Path(row["frame_file"]).stem, {})
    if frame.get("status") == "uncertain":
        return None, False
    if frame.get("status") == "absent":
        return 0, False
    if row["source"] == "hpwren" and state["normal_status"] == "confirmed_normal" and frame.get("status") != "visible":
        return 0, False
    return None, False


selected = {}
arrays = {}
for split in ("train", "val", "test"):
    selected[split] = [r for r in ROWS if r["split"] == split and r["feature_file"] and
                       (split == "test" or annotation(r)[0] in (0, 1)) and
                       (r["source"] != "hpwren" or LABELS["sequences"][r["sequence"]]["normal_status"] == "confirmed_normal")]
    rows = selected[split]
    arrays[split] = (torch.tensor(np.stack([current_features(r, CFG["feature_dimension"]) for r in rows])),
                     torch.tensor(np.stack([base_features(r) for r in rows])))

train = selected["train"]
train_y = torch.tensor([annotation(r)[0] for r in train])
sizes = Counter(r["sequence"] for r in train)
weights = torch.tensor([1 / sizes[r["sequence"]] for r in train], dtype=torch.float64)
pos_weight = (weights[train_y == 0].sum() / weights[train_y == 1].sum()).float()
result = {
    "scope": "Post-hoc read-only diagnosis of saved best checkpoints. Not a new experiment or held-out causal ablation.",
    "input_sha256": {},
    "training_sampling": {
        "candidates": len(train), "class_counts": dict(Counter(train_y.tolist())),
        "sequences": len(sizes), "expected_positive_fraction": float(weights[train_y == 1].sum() / weights.sum()),
        "bce_pos_weight": float(pos_weight),
        "independent_normal_sequence_probability": float(sum(1 / len(sizes) for name in sizes if name.startswith("normal_"))),
        "event_weighted_early_smoke_share": float(weights[(train_y == 1) & torch.tensor([0 <= r["offset_seconds"] <= 300 for r in train])].sum() / weights[train_y == 1].sum()),
        "event_weighted_late_smoke_share": float(weights[(train_y == 1) & torch.tensor([r["offset_seconds"] > 600 for r in train])].sum() / weights[train_y == 1].sum()),
        "positive_offsets": {name: sum(annotation(r)[0] == 1 and predicate(r["offset_seconds"] / 60) for r in train)
                             for name, predicate in (("before_onset", lambda t: t < 0),
                                                     ("0_to_5_minutes", lambda t: 0 <= t <= 5),
                                                     ("5_to_10_minutes", lambda t: 5 < t <= 10),
                                                     ("over_10_minutes", lambda t: t > 10))},
    },
    "models": [], "bes_geometry": [],
}
original_files = [EXP / "protocol.json", EXP / "scripts" / "models7.py", EXP / "scripts" / "learning7.py",
                  DATA / "labels.json", HERE / "analysis.json"]
for seed in (42, 43, 44):
    directory = RUN / f"seed_{seed}"
    predictions = {r["id"]: r for r in (json.loads(line) for line in (directory / "test_predictions.jsonl").read_text().splitlines())}
    thresholds = read(directory / "thresholds.json")["variants"]
    original_files += [directory / "thresholds.json", directory / "test_predictions.jsonl"]
    generator = torch.Generator().manual_seed(seed)
    indices = torch.multinomial(weights, len(train), replacement=True, generator=generator)
    for variant in ("C0", "P1", "P2", "P3", "P4", "P5"):
        path = directory / f"{variant}.pt"
        original_files += [path, directory / f"{variant}_history.json"]
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = RegionalHead(variant, checkpoint["config"], seed)
        model.load_state_dict(checkpoint["state"])
        model.eval()
        x, base = arrays["test"]
        base = base.clone().requires_grad_(True)
        logits, _, appearance = model(x, base)
        probabilities = logits.sigmoid().detach()
        slope = torch.autograd.grad(logits.sum(), base)[0][:, 0] / 10
        stored = torch.tensor([predictions[r["id"]]["scores"][variant] for r in selected["test"]])
        error = float((stored - probabilities).abs().max())
        assert error < 2e-5, (variant, seed, error)
        history = read(directory / f"{variant}_history.json")
        best = history[checkpoint["best_epoch"] - 1]
        threshold = thresholds[variant]["threshold"]
        normal_idx = [i for i, r in enumerate(selected["test"]) if r["source"] == "hpwren"]
        fp_idx = [i for i in normal_idx if probabilities[i] >= threshold]
        facts = {
            "seed": seed, "variant": variant, "threshold": threshold, "best_epoch": checkpoint["best_epoch"],
            "parameters": checkpoint["trainable_parameters"], "best_val_bce": best["val_event_bce"],
            "last_val_bce": history[-1]["val_event_bce"], "best_train_total_loss": best["train_loss"],
            "last_train_total_loss": history[-1]["train_loss"], "max_score_reproduction_error": error,
            "base_logit_derivative": {"min": float(slope.min()), "median": float(slope.median()),
                                       "max": float(slope.max()), "negative_count": int((slope < 0).sum())},
            "normal_candidate_false_positives": len(fp_idx),
            "normal_false_positive_old_box_count": sum(selected["test"][i]["age_seconds"] > 120 for i in fp_idx),
            "appearance_only_normal_fp_at_half": int((appearance.detach().argmax(1)[normal_idx] == 1).sum()),
        }
        by_sequence = defaultdict(list)
        for i in fp_idx:
            row = selected["test"][i]
            by_sequence[row["sequence"]].append({"id": row["id"], "base_probability": row["official_probability"],
                "new_probability": float(probabilities[i]), "age_seconds": row["age_seconds"]})
        facts["normal_false_positives_by_sequence"] = dict(by_sequence)
        result["models"].append(facts)
        if variant not in ("P3", "P5"):
            continue
        batches = []
        for start in range(0, len(indices), CFG["batch_size"]):
            idx = indices[start:start + CFG["batch_size"]]
            x, base = (a[idx] for a in arrays["train"])
            y = train_y[idx]
            logits, z, appearance = model(x, base)
            assignments, _, _ = assign_to_prototypes(z, y, model.prototypes, CFG)
            total, within, occupied = scatter_matrices(z, model.prototypes, assignments)
            lower = torch.linalg.cholesky((within + within.t()) / 2 + CFG["scatter_ridge"] * torch.eye(z.shape[-1]))
            left = torch.linalg.solve_triangular(lower, total - within, upper=False)
            whitened = torch.linalg.solve_triangular(lower, left.t(), upper=False).t()
            eig = torch.linalg.eigvalsh((whitened + whitened.t()) / 2)
            bottom = eig[:CFG["bes_bottom_k"]]
            geometry, _ = bes_loss(z, model.prototypes, assignments, CFG)
            supervised = F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=pos_weight) + CFG["appearance_loss_weight"] * F.cross_entropy(appearance, y)
            g1 = torch.autograd.grad(supervised, z, retain_graph=True)[0].detach().flatten()
            g2 = torch.autograd.grad(CFG["bes_weight"] * geometry, z)[0].detach().flatten()
            batches.append({
                "n": len(idx), "unique_candidates": len(set(idx.tolist())), "smoke": int(y.sum()), "occupied_groups": occupied,
                "bottom_eigenvalues": bottom.detach().tolist(), "bottom_negative": int((bottom < -1e-6).sum()),
                "all_negative_eigenvalues": int((eig < -1e-6).sum()), "bes": float(geometry.detach()),
                "weighted_bes_gradient_over_supervised_gradient": float(g2.norm() / g1.norm().clamp_min(1e-12)),
                "bes_supervised_gradient_cosine": float(F.cosine_similarity(g1, g2, dim=0)),
            })
        result["bes_geometry"].append({"seed": seed, "variant": variant, "checkpoint_epoch": checkpoint["best_epoch"], "batches": batches})

for path in original_files:
    result["input_sha256"][str(path.relative_to(EXP.parents[1]))] = digest(path)
result["prototype_assignments"] = read(HERE / "analysis.json")["prototype_diagnostics"]
result["metric_summary"] = read(HERE / "analysis.json")["variants"]
result["early_decision_examples"] = []
for seed in (42, 43, 44):
    directory = RUN / f"seed_{seed}"
    thresholds = read(directory / "thresholds.json")["variants"]
    for line in (directory / "test_predictions.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["sequence"] == "20261015_HendersonFire_buff-s-mobo-c" and row["offset_seconds"] == 120:
            result["early_decision_examples"].append({"seed": seed, "id": row["id"], "annotation": annotation(row),
                "R0_score": row["scores"]["R0"], "R0_threshold": thresholds["R0"]["threshold"],
                "P5_score": row["scores"]["P5"], "P5_threshold": thresholds["P5"]["threshold"]})
all_batches = [batch for model in result["bes_geometry"] for batch in model["batches"]]
result["bes_summary"] = {
    "sampling_scope": "One sequence-weighted set of minibatches per seed, evaluated at saved P3/P5 best checkpoints; not the historical training trajectory.",
    "batches": len(all_batches), "all_eight_bottom_eigenvalues_negative": sum(b["bottom_negative"] == 8 for b in all_batches),
    "negative_bottom_eigenvalues": sum(b["bottom_negative"] for b in all_batches),
    "median_weighted_bes_gradient_norm_over_supervised": float(np.median([b["weighted_bes_gradient_over_supervised_gradient"] for b in all_batches])),
}
output = HERE / "psl_failure_diagnosis.json"
if "--stdout" in sys.argv:
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    raise SystemExit(0)
output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps({"saved": str(output), "sampling": result["training_sampling"],
    "score_reproduction_max_error": max(r["max_score_reproduction_error"] for r in result["models"]),
    "bes": [{"seed": r["seed"], "variant": r["variant"],
              "negative_bottom8_min": min(b["bottom_negative"] for b in r["batches"]),
              "negative_bottom8_max": max(b["bottom_negative"] for b in r["batches"]),
              "median_gradient_ratio": float(np.median([b["weighted_bes_gradient_over_supervised_gradient"] for b in r["batches"]])),
              "median_gradient_cosine": float(np.median([b["bes_supervised_gradient_cosine"] for b in r["batches"]]))}
             for r in result["bes_geometry"]]}, ensure_ascii=False))
