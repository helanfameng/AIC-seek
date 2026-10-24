"""Event-balanced training, validation selection, and identical score fallback."""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import common7 as c
from data7 import annotation, readiness
from models7 import RegionalHead, assign_to_prototypes, bes_loss, puo_update, scatter_matrices, supervised_contrastive


def base_features(row):
    return np.array([row["logit"] / 10, np.log1p(row["length"]),
                     row["mean_confidence"], row["n_tubes"] / 10], dtype=np.float32)


def current_features(row, dimension):
    with np.load(row["feature_file"], allow_pickle=False) as stored:
        features = stored["features"].astype(np.float32)
    if features.shape != (8, 16, dimension) or not np.isfinite(features).all():
        raise ValueError(f"Unexpected frozen feature tensor: {row['id']}: {features.shape}")
    return features[-1].mean(0)


def tensors(rows, labels, split, cfg, smoke=False):
    selected = [(r, annotation(r, labels)[1]) for r in rows
                if r["split"] == split and r["feature_file"] and
                (r["source"] != "hpwren" or labels["sequences"][r["sequence"]].get("normal_status") == "confirmed_normal")]
    selected = [(r, y) for r, y in selected if y in (0, 1)]
    if smoke:
        # Balanced round-robin across sequence/class groups, never test data.
        buckets = defaultdict(list)
        for item in selected:
            buckets[(item[0]["sequence"], item[1])].append(item)
        selected = []
        while buckets and len(selected) < 128:
            for key in list(buckets):
                selected.append(buckets[key].pop(0))
                if not buckets[key]:
                    del buckets[key]
                if len(selected) >= 128:
                    break
    counts = Counter(y for _, y in selected)
    if not counts[0] or not counts[1]:
        raise ValueError(f"Both reviewed classes required in {split}: {dict(counts)}")
    event_names = sorted({r["sequence"] for r, _ in selected})
    ids = {name: i for i, name in enumerate(event_names)}
    x = torch.tensor(np.stack([current_features(r, cfg["feature_dimension"]) for r, _ in selected]))
    base = torch.tensor(np.stack([base_features(r) for r, _ in selected]))
    y = torch.tensor([y for _, y in selected], dtype=torch.long)
    events = torch.tensor([ids[r["sequence"]] for r, _ in selected], dtype=torch.long)
    dataset = TensorDataset(x, base, y, events)
    return dataset, {"candidates": len(selected), "class_counts": {str(k): int(v) for k, v in counts.items()}, "events": event_names,
                     "candidate_ids_sha256": c.object_digest([r["id"] for r, _ in selected])}


def validation_loss(model, loader, device):
    model.eval()
    by_event = defaultdict(list)
    with torch.inference_mode():
        for x, base, y, events in loader:
            logits, _, _ = model(x.to(device), base.to(device))
            loss = F.binary_cross_entropy_with_logits(logits, y.to(device).float(), reduction="none").cpu()
            for event, value in zip(events.tolist(), loss.tolist()):
                by_event[event].append(value)
    return float(np.mean([np.mean(v) for v in by_event.values()]))


def fit(model, train, val, cfg, device, seed):
    model.to(device)
    train_x, _, train_y, train_events = train.tensors
    model.initialize_prototypes(train_x.to(device), train_y.to(device))
    sizes = torch.bincount(train_events)
    weights = 1.0 / sizes[train_events].double()
    positive = weights[train_y == 1].sum()
    negative = weights[train_y == 0].sum()
    pos_weight = (negative / positive).to(device).float()
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, len(train), replacement=True, generator=generator)
    loader = DataLoader(train, batch_size=cfg["batch_size"], sampler=sampler, num_workers=0)
    validation = DataLoader(val, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    best, best_state, best_epoch, stale, history = float("inf"), None, 0, 0, []
    for epoch in range(cfg["epochs"]):
        model.train()
        stats = defaultdict(float)
        for x, base, y, events in loader:
            x, base, y, events = [v.to(device) for v in (x, base, y, events)]
            logits, z, appearance = model(x, base)
            classification = F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=pos_weight)
            auxiliary = F.cross_entropy(appearance, y)
            loss = classification + cfg["appearance_loss_weight"] * auxiliary
            assignments = means = None
            if model.prototypes is not None:
                assignments, means, _ = assign_to_prototypes(z, y, model.prototypes, cfg)
            if model.variant in ("P3", "P5"):
                scatter_loss, diagnostics = bes_loss(z, model.prototypes, assignments, cfg)
                loss = loss + cfg["bes_weight"] * scatter_loss
                stats["bes_active_batches"] += diagnostics["bes_active"]
                stats["bes_loss"] += float(scatter_loss.detach())
                stats["max_scatter_condition"] = max(stats["max_scatter_condition"], diagnostics.get("scatter_condition", 0))
            if model.variant == "C1":
                loss = loss + cfg["supcon_weight"] * supervised_contrastive(z, y, events, cfg["temperature"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss in {model.variant}, epoch {epoch + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            model.normalize_prototypes()
            if model.variant in ("P4", "P5"):
                total, _, _ = scatter_matrices(z.detach(), model.prototypes.detach(), assignments)
                puo_update(model.prototypes, means, total, cfg)
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite model parameters")
            stats["train_loss_sum"] += float(loss.detach()) * len(x)
            stats["train_examples"] += len(x)
            stats["batches"] += 1
            stats["max_gradient_norm"] = max(stats["max_gradient_norm"], float(gradient_norm))
        measured = validation_loss(model, validation, device)
        row = {"epoch": epoch + 1, "train_loss": stats["train_loss_sum"] / stats["train_examples"],
               "val_event_bce": measured, "batches": int(stats["batches"]),
               "bes_active_batches": int(stats["bes_active_batches"]),
               "bes_loss_mean": stats["bes_loss"] / stats["batches"],
               "max_gradient_norm": stats["max_gradient_norm"],
               "max_scatter_condition": stats["max_scatter_condition"]}
        history.append(row)
        print(model.variant, epoch + 1, "train", round(row["train_loss"], 6), "val", round(measured, 6), flush=True)
        if measured < best:
            best, stale, best_epoch = measured, 0, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break
    if best_state is None:
        raise ValueError("No finite validation checkpoint")
    model.load_state_dict(best_state)
    return history, best_epoch


@torch.inference_mode()
def prototype_diagnostics(model, dataset, info, device):
    if model.prototypes is None:
        return {"kind": "ordinary_classifier_no_prototypes"}
    model.eval()
    x, base, labels, events = dataset.tensors
    _, z, _ = model(x.to(device), base.to(device))
    p = F.normalize(model.prototypes, dim=-1)
    similarities = torch.einsum("bd,ckd->bck", z, p).cpu()
    per_class = {}
    for label in (0, 1):
        selected = labels == label
        nearest = similarities[selected, label].argmax(1)
        event_ids = events[selected]
        k = p.shape[1]
        per_class[str(label)] = {
            "nearest_counts": torch.bincount(nearest, minlength=k).tolist(),
            "per_event": {name: torch.bincount(nearest[event_ids == i], minlength=k).tolist()
                          for i, name in enumerate(info["events"]) if (event_ids == i).any()}}
    flat = p.flatten(0, 1)
    return {"kind": "unconstrained_nearest_prototype_within_reviewed_class_not_sinkhorn_balanced",
            "per_class": per_class, "cosine_gram": (flat @ flat.t()).cpu().tolist()}


def train(args, smoke=False):
    rows = readiness(args)
    cfg = c.protocol()
    if smoke:
        cfg = dict(cfg, epochs=2, patience=2)
    annotations = c.labels()
    train_data, train_info = tensors(rows, annotations, "train", cfg, smoke)
    val_data, val_info = tensors(rows, annotations, "val", cfg, smoke)
    variants = list(args.variants or cfg["variants"])
    common = {"config": cfg, "seed": args.seed, "protocol_sha256": c.digest(c.EXPERIMENT / "protocol.json"),
              "manifest_selection_sha256": c.digest(c.EXPERIMENT / "data_selection.json"),
              "code_fingerprints": c.code_fingerprints(),
              "cache_identity": c.object_digest(c.require_cache(args)),
              "label_fingerprints": {s: c.fingerprint(s, annotations) for s in ("train", "val")},
              "training_data": train_info, "validation_data": val_info,
              "variants": variants, "smoke_test_only": smoke,
              "torch_version": str(torch.__version__)}
    signature = c.object_digest(common)[:12]
    directory = c.folder(args) / (f"smoke/{signature}" if smoke else f"seed_{args.seed}")
    directory.mkdir(parents=True, exist_ok=True)
    plan = directory / "training_plan.json"
    if plan.exists():
        if c.read_json(plan) != common:
            raise ValueError("Run training identity changed; choose a new RunId")
    else:
        c.write_json(plan, common)
    for variant in variants:
        checkpoint = directory / f"{variant}.pt"
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            c.assert_training_identity(saved, args)
            print("Completed checkpoint reused:", variant, flush=True)
            continue
        c.seed_everything(args.seed)
        model = RegionalHead(variant, cfg, args.seed)
        try:
            history, best_epoch = fit(model, train_data, val_data, cfg, args.device, args.seed)
        except Exception as error:
            c.write_json(directory / f"{variant}.failed.json", {"created_at": c.now(), "error": str(error)})
            raise
        state = dict(common, variant=variant, best_epoch=best_epoch, created_at=c.now(),
                     trainable_parameters=sum(p.numel() for p in model.parameters()),
                     state={k: v.detach().cpu() for k, v in model.state_dict().items()})
        temporary = checkpoint.with_suffix(".pt.tmp")
        torch.save(state, temporary)
        temporary.replace(checkpoint)
        c.write_json(directory / f"{variant}_history.json", history)
        c.write_json(directory / f"{variant}_prototypes.json", {
            "train": prototype_diagnostics(model, train_data, train_info, args.device),
            "val": prototype_diagnostics(model, val_data, val_info, args.device)})
    if smoke:
        c.write_json(c.folder(args) / "smoke" / "latest.json", {
            "directory": str(directory), "signature": signature, "variants": variants,
            "seed": args.seed, "smoke_test_only": True})
    print("Smoke compatibility completed" if smoke else "Training completed", directory, flush=True)


def scores(args, rows):
    directory = c.folder(args) / f"seed_{args.seed}"
    plan = c.read_json(directory / "training_plan.json")
    if plan.get("smoke_test_only"):
        raise ValueError("Smoke tests cannot produce evaluation results")
    result = {r["id"]: {"R0": float(r["official_probability"])} for r in rows}
    usable = [r for r in rows if r["feature_file"]]
    timing = {"scope": "new_head_only_excludes_feature_IO_model_loading_detector_and_encoder",
              "device": args.device, "candidates": len(rows), "feature_candidates": len(usable), "variants": {}}
    x = torch.tensor(np.stack([current_features(r, plan["config"]["feature_dimension"]) for r in usable])) if usable else None
    base = torch.tensor(np.stack([base_features(r) for r in usable])) if usable else None
    for variant in plan["variants"]:
        state = torch.load(directory / f"{variant}.pt", map_location="cpu", weights_only=True)
        c.assert_training_identity(state, args)
        if state.get("smoke_test_only"):
            raise ValueError("Smoke checkpoint cannot be evaluated")
        model = RegionalHead(variant, state["config"], state["seed"]).to(args.device).eval()
        model.load_state_dict(state["state"])
        # All new heads use exactly the same fallback, including C0/C1.
        for row in rows:
            result[row["id"]][variant] = float(row["official_probability"])
        if args.device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(usable), 128):
                value, _, _ = model(x[start:start+128].to(args.device), base[start:start+128].to(args.device))
                for row, probability in zip(usable[start:start+128], value.sigmoid().cpu().tolist()):
                    result[row["id"]][variant] = float(probability)
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        timing["variants"][variant] = {"forward_seconds": elapsed,
            "milliseconds_per_feature_candidate": elapsed * 1000 / len(usable) if usable else None}
    split_names = sorted({r["split"] for r in rows})
    c.write_json(directory / ("score_timing_" + "_".join(split_names) + ".json"), timing)
    return result, ["R0", *plan["variants"]]
