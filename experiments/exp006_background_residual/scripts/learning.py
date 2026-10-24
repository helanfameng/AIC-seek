"""Train only new modules on frozen official features; never update Pyronear."""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from common import DATA, EXPERIMENT, configure_cache, digest, load_manifest, now, protocol, read_json, read_jsonl, seed_everything, write_json
from features import history_indices

VARIANTS = ("calibration_only", "history", "mean_residual", "learned_residual")


def load_records(folder):
    meta = read_json(folder / "run.json")
    result = []
    for seq_id in meta["sequence_ids"]:
        result.extend(read_jsonl(folder / f"{seq_id}.jsonl"))
    return result


def require_full_run(folder):
    meta = read_json(folder / "run.json")
    if meta["partial"] or not meta["complete"]:
        raise ValueError("Training/evaluation requires a complete full-data extraction, not a smoke test")
    if meta["protocol_sha256"] != digest(EXPERIMENT / "protocol.json") or \
       meta["manifest_selection_sha256"] != digest(EXPERIMENT / "data_selection.json"):
        raise ValueError("Protocol/data selection changed after extraction; choose a new run-id")
    return meta


def split_fingerprint(labels, sequences, split):
    state = {s["id"]: labels["sequences"].get(s["id"], {}) for s in sequences if s["split"] == split}
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def normal_fingerprint(labels, sequences, split):
    state = {s["id"]: {key: labels["sequences"][s["id"]].get(key) for key in ("normal_status", "frames")}
             for s in sequences if s["split"] == split and s["source"] == "hpwren"}
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def candidate_label(row, labels):
    from review_assist import annotation
    state = labels["sequences"].get(row["sequence"], {})
    return annotation(row, state)[1]


def base_features(row):
    return np.array([row["logit"] / 10, np.log1p(row["length"]),
                     row["mean_confidence"], row["n_tubes"] / 10], dtype=np.float32)


class BackgroundPredictor(nn.Module):
    def __init__(self, dimension, hidden):
        super().__init__()
        self.gru = nn.GRU(dimension, hidden, batch_first=True)
        self.output = nn.Linear(hidden, dimension)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, history):
        # B,T,G,D -> B*G,T,D. No current or future feature is an input.
        history = F.normalize(history, dim=-1)
        b, t, g, d = history.shape
        _, hidden = self.gru(history.permute(0, 2, 1, 3).reshape(b * g, t, d))
        delta = self.output(hidden[-1]).reshape(b, g, d)
        return F.normalize(history[:, -1] + delta, dim=-1)


class Fusion(nn.Module):
    def __init__(self, variant, dimension, cfg, predictor=None):
        super().__init__()
        self.variant, self.cfg = variant, cfg
        self.predictor = predictor
        hidden = cfg["hidden_dim"]
        if predictor is not None:
            for parameter in predictor.parameters():
                parameter.requires_grad_(False)
        if variant == "calibration_only":
            self.head = nn.Linear(4, 1)
        elif variant == "history":
            self.gru = nn.GRU(dimension, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(2 * hidden + 4, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        else:
            self.head = nn.Sequential(nn.Linear((dimension + 2) * cfg["residual_steps"] + 4, hidden),
                                      nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, features, base):
        if self.variant == "calibration_only":
            return self.head(base).squeeze(-1)
        features = F.normalize(features, dim=-1)
        if self.variant == "history":
            b, t, g, d = features.shape
            _, hidden = self.gru(features.permute(0, 2, 1, 3).reshape(b * g, t, d))
            hidden = hidden[-1].reshape(b, g, -1)
            summary = torch.cat([hidden.mean(1), hidden.amax(1)], -1)
        else:
            parts = []
            k = self.cfg["history_frames"]
            for index in range(k, k + self.cfg["residual_steps"]):
                history = features[:, index - k:index]
                if self.variant == "mean_residual":
                    reference = F.normalize(history.mean(1), dim=-1)
                else:
                    with torch.no_grad():
                        reference = self.predictor(history)
                residual = features[:, index] - reference
                norms = residual.norm(dim=-1)
                parts.extend([residual.mean(1), norms.mean(1, keepdim=True), norms.amax(1, keepdim=True)])
            summary = torch.cat(parts, -1)
        return self.head(torch.cat([summary, base], -1)).squeeze(-1)


class NormalWindows(Dataset):
    def __init__(self, folder, sequences, labels, split, cfg):
        self.banks, self.samples = [], []
        self.k = cfg["history_frames"]
        for seq in sequences:
            state = labels["sequences"].get(seq["id"], {})
            if seq["source"] != "hpwren" or seq["split"] != split or state.get("normal_status") != "confirmed_normal":
                continue
            bank = np.load(folder / "features" / f"normal_{seq['id']}.npz")["features"]
            bank_index = len(self.banks)
            self.banks.append(bank)
            for end in range(self.k, len(seq["frames"])):
                ids = history_indices(seq["frames"], end, self.k + 1, cfg["maximum_history_gap_seconds"])
                if ids is None:
                    continue
                if any(state.get("frames", {}).get(Path(seq["frames"][i]["file"]).stem, {}).get("status")
                       in ("visible", "uncertain") for i in ids):
                    continue
                for region in range(bank.shape[1]):
                    self.samples.append((bank_index, region, end))
        if not self.samples:
            raise ValueError(f"No reviewed normal {split} windows. Run Prepare/Review and confirm complete normal clips first.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        bank_index, region, end = self.samples[index]
        value = self.banks[bank_index][end - self.k:end + 1, region].astype(np.float32)
        return torch.from_numpy(value[:-1]), torch.from_numpy(value[-1])


class CandidateWindows(Dataset):
    def __init__(self, folder, rows, labels, split):
        self.folder = folder
        self.rows = [(row, candidate_label(row, labels)) for row in rows
                     if row["split"] == split and row["feature_file"] is not None]
        self.rows = [(row, label) for row, label in self.rows if label is not None]
        counts = Counter(label for _, label in self.rows)
        if not counts[0] or not counts[1]:
            raise ValueError(f"Reviewed {split} candidates need both smoke and background, found {dict(counts)}. Use Review.")
        self.counts = dict(counts)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row, label = self.rows[index]
        with np.load(self.folder / row["feature_file"]) as data:
            features = torch.from_numpy(data["features"].astype(np.float32))
        return features, torch.from_numpy(base_features(row)), torch.tensor(float(label))


def fit_loop(model, train_data, val_data, cfg, device, kind):
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    train_loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    epochs = cfg["predictor_epochs"] if kind == "predictor" else cfg["fusion_epochs"]
    if kind != "predictor":
        pos_weight = torch.tensor(train_data.counts[0] / train_data.counts[1], device=device)
    best, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        losses = {}
        for part, loader in (("train", train_loader), ("val", val_loader)):
            model.train(part == "train")
            total, count = 0.0, 0
            with torch.set_grad_enabled(part == "train"):
                for batch in loader:
                    batch = [x.to(device) for x in batch]
                    if kind == "predictor":
                        x, target = batch
                        loss = (1 - (model(x) * F.normalize(target, dim=-1)).sum(-1)).mean()
                    else:
                        x, base, target = batch
                        loss = F.binary_cross_entropy_with_logits(model(x, base), target, pos_weight=pos_weight)
                    if part == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    total += loss.item() * len(batch[0])
                    count += len(batch[0])
            losses[part] = total / count
        history.append(dict(epoch=epoch + 1, **losses))
        print(kind, epoch + 1, losses, flush=True)
        if losses["val"] < best:
            best, stale = losses["val"], 0
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
        if stale >= cfg["early_stopping_patience"]:
            break
    model.load_state_dict(best_state)
    return history


def train(args):
    configure_cache()
    seed_everything(args.seed)
    cfg = protocol()
    folder = EXPERIMENT / "outputs" / args.run_id / args.setting
    meta = require_full_run(folder)
    labels = read_json(DATA / "labels.json")
    sequences = load_manifest()["sequences"]
    checkpoints = folder / f"seed_{args.seed}"
    checkpoints.mkdir(exist_ok=True)
    common = {"config": cfg, "seed": args.seed, "created_at": now(), "run_identity": meta,
              "label_fingerprints": {s: split_fingerprint(labels, sequences, s) for s in ("train", "val")}}
    normal_labels = {s: normal_fingerprint(labels, sequences, s) for s in ("train", "val")}
    if args.stage in ("predictor", "train"):
        if (checkpoints / "predictor.pt").exists():
            raise ValueError("Predictor checkpoint already exists; use a new run/seed, or Fusion to continue after Predictor")
        train_data = NormalWindows(folder, sequences, labels, "train", cfg)
        val_data = NormalWindows(folder, sequences, labels, "val", cfg)
        dimension = train_data[0][0].shape[-1]
        model = BackgroundPredictor(dimension, cfg["hidden_dim"])
        history = fit_loop(model, train_data, val_data, cfg, args.device, "predictor")
        torch.save(dict(common, normal_label_fingerprints=normal_labels, dimension=dimension, state=model.cpu().state_dict(),
                        train_windows=len(train_data), val_windows=len(val_data)), checkpoints / "predictor.pt")
        write_json(checkpoints / "predictor_history.json", history)
    if args.stage in ("fusion", "train"):
        if any((checkpoints / f"{v}.pt").exists() for v in VARIANTS):
            raise ValueError("Fusion checkpoints already exist; preserve them and choose a new run/seed")
        state = torch.load(checkpoints / "predictor.pt", map_location="cpu", weights_only=True)
        if state["normal_label_fingerprints"] != normal_labels:
            raise ValueError("Normal train/val labels changed since predictor training; use a new run/seed")
        dimension = state["dimension"]
        rows = load_records(folder)
        train_data = CandidateWindows(folder, rows, labels, "train")
        val_data = CandidateWindows(folder, rows, labels, "val")
        for variant in VARIANTS:
            seed_everything(args.seed)
            predictor = None
            if variant == "learned_residual":
                predictor = BackgroundPredictor(dimension, cfg["hidden_dim"])
                predictor.load_state_dict(state["state"])
            model = Fusion(variant, dimension, cfg, predictor)
            history = fit_loop(model, train_data, val_data, cfg, args.device, variant)
            torch.save(dict(common, dimension=dimension, variant=variant, state=model.cpu().state_dict(),
                            train_candidates=len(train_data), val_candidates=len(val_data),
                            class_counts=train_data.counts,
                            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)),
                       checkpoints / f"{variant}.pt")
            write_json(checkpoints / f"{variant}_history.json", history)


def score_rows(folder, seed, device, rows):
    checkpoint_folder = folder / f"seed_{seed}"
    result = {row["id"]: {"original": row["official_probability"]} for row in rows}
    for variant in VARIANTS:
        state = torch.load(checkpoint_folder / f"{variant}.pt", map_location="cpu", weights_only=True)
        cfg, dimension = state["config"], state["dimension"]
        predictor = BackgroundPredictor(dimension, cfg["hidden_dim"]) if variant == "learned_residual" else None
        model = Fusion(variant, dimension, cfg, predictor).to(device).eval()
        model.load_state_dict(state["state"])
        with torch.inference_mode():
            for row in rows:
                if row["feature_file"] is None:
                    probability = row["official_probability"]
                else:
                    with np.load(folder / row["feature_file"]) as data:
                        x = torch.from_numpy(data["features"].astype(np.float32))[None].to(device)
                    base = torch.from_numpy(base_features(row))[None].to(device)
                    probability = float(torch.sigmoid(model(x, base)).item())
                result[row["id"]][variant] = probability
    return result
