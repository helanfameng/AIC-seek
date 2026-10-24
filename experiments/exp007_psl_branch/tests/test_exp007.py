"""Numerical, leakage, metric, and review-state regressions for experiment seven."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

import common7 as c
from legacy7 import bind
from models7 import RegionalHead, assign_to_prototypes, balanced_assignments, bes_loss, puo_update, supervised_contrastive


class Numerics(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.cfg = c.protocol()

    def test_shared_modules_have_identical_initialization(self):
        a, b = RegionalHead("C0", self.cfg), RegionalHead("P5", self.cfg)
        for module in ("projector", "fusion"):
            for x, y in zip(getattr(a, module).parameters(), getattr(b, module).parameters()):
                self.assertTrue(torch.equal(x, y))

    def test_all_variants_can_backpropagate(self):
        for variant in ("C0", "P1", "P2", "P3", "P4", "P5", "C1"):
            with self.subTest(variant=variant):
                model = RegionalHead(variant, self.cfg)
                x, base = torch.randn(64, 384), torch.randn(64, 4)
                y = torch.arange(64) % 2
                logit, z, appearance = model(x, base)
                loss = F.binary_cross_entropy_with_logits(logit, y.float()) + F.cross_entropy(appearance, y)
                if variant in ("P3", "P5"):
                    assignments, _, _ = assign_to_prototypes(z, y, model.prototypes, self.cfg)
                    loss = loss + self.cfg["bes_weight"] * bes_loss(z, model.prototypes, assignments, self.cfg)[0]
                if variant == "C1":
                    loss = loss + supervised_contrastive(z, y, torch.arange(64) // 4)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_sinkhorn_balances_nontrivial_counts(self):
        similarity = torch.randn(80, 4) * 0.1
        q = balanced_assignments(similarity, epsilon=0.3, iterations=50)
        self.assertTrue(torch.allclose(q.sum(1), torch.ones(80), atol=1e-5))
        self.assertTrue(torch.allclose(q.sum(0), torch.full((4,), 20.0), atol=1e-3))

    def test_assignments_do_not_cross_ground_truth_class(self):
        z = F.normalize(torch.randn(30, 32), dim=-1)
        y = torch.arange(30) % 2
        p = F.normalize(torch.randn(2, 2, 32), dim=-1)
        assigned, means, valid = assign_to_prototypes(z, y, p, self.cfg)
        self.assertTrue(torch.equal(assigned // 2, y))
        self.assertTrue(torch.isfinite(means).all())

    def test_singular_scatter_does_not_crash(self):
        z = torch.zeros(16, 32, requires_grad=True)
        p = torch.zeros(2, 2, 32, requires_grad=True)
        a = torch.arange(16) % 4
        loss, details = bes_loss(z, p, a, self.cfg)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_bes_requires_at_least_two_occupied_groups(self):
        z = torch.randn(4, 32, requires_grad=True)
        p = torch.randn(2, 2, 32, requires_grad=True)
        loss, details = bes_loss(z, p, torch.zeros(4, dtype=torch.long), self.cfg)
        self.assertEqual(details["bes_active"], 0)
        self.assertEqual(float(loss), 0)

    def test_puo_produces_uncorrelated_metric_columns(self):
        p = F.normalize(torch.randn(2, 2, 32), dim=-1)
        means = F.normalize(torch.randn_like(p), dim=-1)
        matrix = torch.randn(32, 32)
        scatter = matrix @ matrix.t() / 32
        metric = puo_update(p, means, scatter, self.cfg)
        flat = p.flatten(0, 1)
        gram = flat @ metric @ flat.t()
        off = gram - torch.diag(gram.diag())
        self.assertLess(float(off.abs().max()), 1e-5)
        self.assertTrue(torch.allclose(p.norm(dim=-1), torch.ones(2, 2), atol=1e-5))

    def test_supcon_no_cross_event_positive_returns_zero(self):
        z = F.normalize(torch.randn(4, 8), dim=-1).requires_grad_()
        value = supervised_contrastive(z, torch.tensor([0, 0, 1, 1]), torch.zeros(4, dtype=torch.long))
        value.backward()
        self.assertEqual(float(value), 0)
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_supcon_mixed_isolated_anchor_has_finite_gradients(self):
        z = F.normalize(torch.randn(5, 8), dim=-1).requires_grad_()
        value = supervised_contrastive(z, torch.tensor([0, 0, 0, 1, 1]), torch.tensor([0, 1, 0, 0, 0]))
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(z.grad).all())


class DataAndMetrics(unittest.TestCase):
    def test_history_rejects_future_order_and_large_gaps(self):
        history = bind()["features"].history_indices
        self.assertIsNone(history([{"timestamp": t} for t in (0, 60, 59)], 2, 3, 120))
        self.assertIsNone(history([{"timestamp": t} for t in (0, 60, 300)], 2, 3, 120))
        self.assertEqual(history([{"timestamp": t} for t in (0, 60, 120)], 2, 3, 120), [0, 1, 2])

    def test_no_candidate_frames_still_contribute_normal_hours(self):
        old = bind()["evaluation"]
        sequences = [{"id": "normal", "source": "hpwren", "frames": [
            {"timestamp": t, "file": f"{t}.jpg"} for t in (0, 60, 120)], "end_utc_exclusive": 180}]
        labels = {"sequences": {"normal": {"normal_status": "confirmed_normal", "frames": {}}}}
        rows = [{"id": "r", "sequence": "normal", "source": "hpwren", "timestamp": 60}]
        result = old.normal_statistics(sequences, labels, rows, {"r": {"x": 0.9}}, "x", 0.5, c.protocol())
        self.assertAlmostEqual(result["observed_camera_hours"], 180 / 3600)
        self.assertAlmostEqual(result["positive_time_fraction"], 1 / 3)
        self.assertEqual(result["false_alarm_episodes"], 1)

    def test_high_background_winner_delays_correct_localization(self):
        old = bind()["evaluation"]
        sequence = {"id": "fire", "source": "figlib", "official_onset": 0,
                    "frames": [{"timestamp": 0}, {"timestamp": 120}]}
        rows = [{"id": i, "sequence": "fire", "timestamp": stamp, "frame_file": f"{stamp}.jpg"}
                for i, stamp in (("target1", 60), ("bg", 60), ("target2", 120))]
        labels = {"sequences": {"fire": {"candidates": {
            "target1": {"label": 1, "matches_target": True}, "bg": {"label": 0},
            "target2": {"label": 1, "matches_target": True}}, "frames": {}}}}
        score = {"target1": {"x": .9}, "bg": {"x": .95}, "target2": {"x": .85}}
        result = old.event_statistics([sequence], labels, rows, score, "x", .5)
        self.assertEqual(result["per_event"][0]["delay_minutes_official"], 2)

    def test_unknown_candidate_is_not_a_negative(self):
        annotation = bind()["review_assist"].annotation
        reviewed, label, target = annotation({"id": "r", "frame_file": "0.jpg", "source": "figlib"},
                                              {"frames": {}, "candidates": {}})
        self.assertFalse(reviewed)
        self.assertIsNone(label)

    def test_test_label_changes_do_not_change_train_val_fingerprints(self):
        sandbox = c.ROOT / ".cache" / "exp007" / "tests"
        sandbox.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=sandbox) as directory, patch.object(c, "DATA", Path(directory)):
            c.write_json(c.DATA / "manifest.json", {"sequences": [
                {"id": x, "split": x} for x in ("train", "val", "test")]})
            labels = {"sequences": {x: {"value": 1} for x in ("train", "val", "test")}}
            before = [c.fingerprint(x, labels) for x in ("train", "val")]
            labels["sequences"]["test"] = {"value": 2}
            self.assertEqual(before, [c.fingerprint(x, labels) for x in ("train", "val")])

    def test_eligibility_rejects_invalid_updates_and_backs_up_valid_one(self):
        from review7 import update_eligibility
        sandbox = c.ROOT / ".cache" / "exp007" / "tests"
        sandbox.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=sandbox) as directory, patch.object(c, "DATA", Path(directory)):
            state = {"schema_version": 1, "sequences": {"fire": {"status": "pending"}}}
            path = c.DATA / "eligibility.json"
            c.write_json(path, state)
            before = c.digest(path)
            with self.assertRaises(ValueError):
                update_eligibility({"version": "stale", "sequence": "fire", "status": "eligible"})
            with self.assertRaises(ValueError):
                update_eligibility({"version": c.object_digest(state), "sequence": "fire", "status": "excluded"})
            self.assertEqual(before, c.digest(path))
            update_eligibility({"version": c.object_digest(state), "sequence": "fire", "status": "eligible", "reason": "viewed source frames"})
            self.assertEqual(c.read_json(path)["sequences"]["fire"]["status"], "eligible")
            self.assertEqual(len(list(c.DATA.glob("eligibility.before_session_*.json"))), 1)


class WorkflowIntegration(unittest.TestCase):
    def test_train_calibrate_evaluate_resume_and_tamper_guard(self):
        """Synthetic IO fixture only; never writes real event truth or results."""
        from learning7 import train
        from evaluation7 import calibrate, evaluate, aggregate
        sandbox = c.ROOT / ".cache" / "exp007" / "tests"
        sandbox.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=sandbox) as directory:
            temp = Path(directory)
            with patch.object(c, "DATA", temp / "data"), patch.object(c, "EXPERIMENT", temp / "experiment"), \
                 patch.object(c, "code_fingerprints", return_value={"synthetic_fixture": "fixed"}):
                cfg = c.protocol.__globals__["read_json"](Path(__file__).resolve().parents[1] / "protocol.json")
                cfg.update(epochs=1, patience=1, batch_size=8)
                c.write_json(c.EXPERIMENT / "protocol.json", cfg)
                sequences, states, records = [], {}, {}
                args = SimpleNamespace(run_id="fixture", setting="baseline", seed=42, seeds=[42],
                                       variants=["C0", "P1", "P2", "P3", "P4", "P5", "C1"], device="cpu")
                folder = c.folder(args)
                folder.mkdir(parents=True)
                for split in ("train", "val", "test"):
                    seq_id = split + "_fire"
                    frames = [{"file": f"{seq_id}_{stamp}.jpg", "timestamp": stamp} for stamp in (0, 60, 120, 180)]
                    sequences.append({"id": seq_id, "source": "figlib", "split": split, "site": split,
                                      "official_onset": 0, "frames": frames})
                    states[seq_id] = {"normal_status": "unreviewed", "frames": {}, "human_visible_timestamp": None,
                                      "candidates": {}, "candidate_review_complete": {"baseline": True}}
                    rows = []
                    for index, frame in enumerate(frames):
                        path = folder / f"{seq_id}_{index}.npz"
                        rng = np.random.default_rng(index + len(seq_id))
                        np.savez_compressed(path, features=rng.standard_normal((8, 16, 384)).astype(np.float16))
                        identity = f"{seq_id}/{index}"
                        label = int(index >= 2)
                        rows.append({"id": identity, "sequence": seq_id, "source": "figlib", "split": split,
                                     "timestamp": frame["timestamp"], "frame_file": frame["file"], "feature_file": str(path),
                                     "official_probability": .8 if label else .1, "logit": 1.0 if label else -1.0,
                                     "length": 2, "mean_confidence": .2, "n_tubes": 1})
                        states[seq_id]["candidates"][identity] = {"label": label, "matches_target": bool(label)}
                    records[seq_id] = rows
                    if split != "train":
                        normal_id = split + "_normal"
                        sequences.append({"id": normal_id, "source": "hpwren", "split": split, "site": split,
                            "frames": [{"timestamp": 0, "file": normal_id + ".jpg"}], "end_utc_exclusive": 60})
                        states[normal_id] = {"normal_status": "confirmed_normal", "frames": {}, "candidates": {}}
                        records[normal_id] = [{"id": normal_id, "sequence": normal_id, "source": "hpwren", "split": split,
                            "timestamp": 0, "frame_file": normal_id + ".jpg", "feature_file": None, "official_probability": .2}]
                manifest = {"sequences": sequences}
                c.write_json(c.DATA / "manifest.json", manifest)
                c.write_json(c.EXPERIMENT / "data_selection.json", manifest)
                c.write_json(c.DATA / "labels.json", {"sequences": states})
                c.write_json(c.DATA / "eligibility.json", {"sequences": {"test_fire": {"status": "eligible"}}})
                hashes = {}
                for seq_id, rows in records.items():
                    path = folder / f"{seq_id}.jsonl"
                    c.write_jsonl(path, rows)
                    hashes[seq_id] = c.digest(path)
                feature_hashes = {str(path): c.digest(path) for path in folder.glob("*.npz")}
                c.write_json(folder / "feature_sources.json", feature_hashes)
                c.write_json(folder / "run.json", {"complete": True, "partial": False,
                    "sequence_ids": list(records), "record_sha256": hashes,
                    "protocol_sha256": c.digest(c.EXPERIMENT / "protocol.json"),
                    "manifest_selection_sha256": c.digest(c.EXPERIMENT / "data_selection.json"),
                    "extractor_sha256": c.digest(c.LEGACY / "scripts" / "features.py"),
                    "feature_sources_sha256": c.digest(folder / "feature_sources.json")})
                train(args)
                checkpoint = folder / "seed_42" / "P5.pt"
                before = c.digest(checkpoint)
                train(args)
                self.assertEqual(before, c.digest(checkpoint))
                calibrate(args)
                thresholds = folder / "seed_42" / "thresholds.json"
                frozen = c.digest(thresholds)
                calibrate(args)
                self.assertEqual(frozen, c.digest(thresholds))
                evaluate(args)
                aggregate(args)
                report = c.read_json(folder / "seed_42" / "summary.json")
                self.assertEqual(set(report["variants"]), {"R0", *args.variants})
                self.assertEqual(report["variants"]["R0"]["fire"]["events"], 1)
                self.assertAlmostEqual(report["variants"]["R0"]["normal"]["observed_camera_hours"], 1 / 60)
                # New test truth is allowed after training/calibration but cannot
                # silently overwrite an evaluation that already used old truth.
                states["test_fire"]["human_visible_timestamp"] = 60
                c.write_json(c.DATA / "labels.json", {"sequences": states})
                with self.assertRaises(ValueError):
                    evaluate(args)
                # Train/val identity edits are rejected even before inference.
                states["train_fire"]["candidates"]["train_fire/0"]["label"] = 1
                c.write_json(c.DATA / "labels.json", {"sequences": states})
                with self.assertRaises(ValueError):
                    calibrate(args)


if __name__ == "__main__":
    unittest.main()
