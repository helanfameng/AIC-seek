"""Meaningful guards: causal windows, shared-label semantics, leakage, and metrics."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import check_split_integrity, configure_cache, protocol
configure_cache()
import torch
from features import history_indices
from learning import BackgroundPredictor, Fusion, candidate_label, fit_loop
from evaluation import normal_statistics, event_statistics, ensure_event_review


class TemporalTests(unittest.TestCase):
    def test_history_is_past_and_current_without_duplicates(self):
        frames = [{"timestamp": t * 60} for t in range(12)]
        self.assertEqual(history_indices(frames, 8, 6, 120), [3, 4, 5, 6, 7, 8])
        self.assertIsNone(history_indices(frames, 3, 6, 120))
        frames[7]["timestamp"] = frames[6]["timestamp"]
        self.assertIsNone(history_indices(frames, 8, 6, 120))

    def test_missing_time_is_not_a_fabricated_frame(self):
        frames = [{"timestamp": t} for t in (0, 60, 120, 500, 560, 620)]
        self.assertIsNone(history_indices(frames, 5, 6, 120))

    def test_site_and_event_leakage_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Site leakage"):
            check_split_integrity([{"site": "a", "split": "train"}, {"site": "a", "split": "test"}])
        with self.assertRaisesRegex(ValueError, "Event leakage"):
            check_split_integrity([{"site": "a", "split": "train", "event_group": "fire"},
                                   {"site": "b", "split": "test", "event_group": "fire"}])

    def test_empty_boxes_and_unreviewed_normal_are_unknown(self):
        row = {"sequence": "a", "id": "candidate", "source": "figlib", "frame_file": "images/0.jpg"}
        labels = {"sequences": {"a": {"frames": {"0": {"smoke_boxes": []}}, "candidates": {}}}}
        self.assertIsNone(candidate_label(row, labels))
        row["source"] = "hpwren"
        self.assertIsNone(candidate_label(row, labels))
        labels["sequences"]["a"]["normal_status"] = "confirmed_normal"
        self.assertEqual(candidate_label(row, labels), 0)
        labels["sequences"]["a"]["frames"]["0"]["status"] = "uncertain"
        self.assertIsNone(candidate_label(row, labels))

    def test_predictor_and_fusion_dimensions_and_frozen_branch(self):
        cfg = dict(protocol(), hidden_dim=8, history_frames=3, residual_steps=2)
        predictor = BackgroundPredictor(12, 8)
        history = torch.randn(2, 5, 4, 12)
        result = predictor(history[:, :3])
        self.assertEqual(result.shape, (2, 4, 12))
        # A predictor sees only the supplied past; target changes cannot affect its output.
        old = predictor(history[:, :3]).detach()
        history[:, 3:] += 100
        self.assertTrue(torch.equal(old, predictor(history[:, :3]).detach()))
        for variant in ("calibration_only", "history", "mean_residual", "learned_residual"):
            model = Fusion(variant, 12, cfg, predictor if variant == "learned_residual" else None)
            score = model(history, torch.zeros(2, 4))
            self.assertEqual(score.shape, (2,))
            score.sum().backward()
            if variant == "learned_residual":
                self.assertTrue(all(not p.requires_grad for p in model.predictor.parameters()))

    def test_persistent_false_alarm_is_one_episode_but_has_duration(self):
        cfg = protocol()
        seq = {"id": "n", "source": "hpwren", "end_utc_exclusive": 600,
               "frames": [{"timestamp": i * 60, "file": f"{i}.jpg"} for i in range(10)]}
        labels = {"sequences": {"n": {"normal_status": "confirmed_normal", "frames": {}}}}
        rows = [{"id": str(i), "sequence": "n", "source": "hpwren", "timestamp": i * 60} for i in range(10)]
        scores = {str(i): {"original": 0.9} for i in range(10)}
        value = normal_statistics([seq], labels, rows, scores, "original", 0.5, cfg)
        self.assertEqual(value["false_alarm_episodes"], 1)
        self.assertEqual(value["positive_minutes"], 10)
        self.assertEqual(value["positive_time_fraction"], 1)

    def test_background_alarm_does_not_count_as_early_target_detection(self):
        seq = {"id": "fire", "source": "figlib", "official_onset": 1000, "frames": [{"timestamp": 1600}]}
        rows = [{"id": "bg", "sequence": "fire", "timestamp": 1060, "source": "figlib", "frame_file": "0.jpg"},
                {"id": "smoke", "sequence": "fire", "timestamp": 1300, "source": "figlib", "frame_file": "1.jpg"}]
        labels = {"sequences": {"fire": {"frames": {}, "human_visible_timestamp": 1120,
            "candidates": {"bg": {"label": 0}, "smoke": {"label": 1, "matches_target": True}}}}}
        scores = {"bg": {"original": 0.99}, "smoke": {"original": 0.8}}
        result = event_statistics([seq], labels, rows, scores, "original", 0.5)
        self.assertEqual(result["median_delay_detected_only"], 5)
        self.assertEqual(result["per_event"][0]["delay_minutes_human_visible"], 3)
        rows.append({"id": "bg2", "sequence": "fire", "timestamp": 1300,
                     "source": "figlib", "frame_file": "1.jpg"})
        labels["sequences"]["fire"]["candidates"]["bg2"] = {"label": 0}
        scores["bg2"] = {"original": 0.99}
        # A lower-scoring target in the same window is not the alarm's location.
        result = event_statistics([seq], labels, rows, scores, "original", 0.5)
        self.assertEqual(result["detected"], 0)

    def test_official_evaluation_accepts_optional_human_time_and_keeps_misses(self):
        sequences = [{"id": name, "source": "figlib", "official_onset": 1000,
                      "frames": [{"timestamp": 1300}, {"timestamp": 1600}]}
                     for name in ("smoke", "background_only", "no_candidates")]
        rows = [{"id": "positive", "sequence": "smoke", "timestamp": 1300,
                 "source": "figlib", "frame_file": "positive.jpg"},
                {"id": "negative", "sequence": "background_only", "timestamp": 1300,
                 "source": "figlib", "frame_file": "negative.jpg"}]
        labels = {"sequences": {name: {"candidate_review_complete": {"baseline": True},
            "frames": {}, "candidates": {}, "human_visible_timestamp": None}
            for name in ("smoke", "background_only", "no_candidates")}}
        labels["sequences"]["smoke"]["candidates"]["positive"] = {"label": 1, "matches_target": True}
        labels["sequences"]["background_only"]["candidates"]["negative"] = {"label": 0}
        ensure_event_review(sequences, rows, labels, "baseline")
        scores = {r["id"]: {"original": .9} for r in rows}
        result = event_statistics(sequences, labels, rows, scores, "original", .5)
        self.assertEqual(result["events"], 3)
        self.assertEqual(result["detected"], 1)
        self.assertEqual(result["missed_within_observation"], 2)
        self.assertEqual(result["median_delay_detected_only"], 5)
        self.assertTrue(all(e["delay_minutes_human_visible"] is None for e in result["per_event"]))

    def test_optional_human_time_does_not_bypass_candidate_review(self):
        seq = {"id": "fire", "source": "figlib", "official_onset": 1000,
               "frames": [{"timestamp": 1300}]}
        rows = [{"id": "pending", "sequence": "fire", "source": "figlib", "frame_file": "0.jpg"}]
        state = {"candidate_review_complete": {"baseline": True}, "frames": {}, "candidates": {}}
        labels = {"sequences": {"fire": state}}
        with self.assertRaisesRegex(ValueError, "still unreviewed"):
            ensure_event_review([seq], rows, labels, "baseline")
        state["candidates"]["pending"] = {"label": None}
        ensure_event_review([seq], rows, labels, "baseline")
        state["candidate_review_complete"]["baseline"] = False
        with self.assertRaisesRegex(ValueError, "Complete candidate identity"):
            ensure_event_review([seq], rows, labels, "baseline")

    def test_recorded_human_time_must_belong_to_sequence(self):
        seq = {"id": "fire", "source": "figlib", "official_onset": 1000,
               "frames": [{"timestamp": 1300}]}
        state = {"candidate_review_complete": {"baseline": True}, "frames": {},
                 "candidates": {}, "human_visible_timestamp": 1299}
        labels = {"sequences": {"fire": state}}
        with self.assertRaisesRegex(ValueError, "outside the sequence"):
            ensure_event_review([seq], [], labels, "baseline")
        state["human_visible_timestamp"] = 1300
        ensure_event_review([seq], [], labels, "baseline")

    def test_training_loop_runs_without_updating_official_weights(self):
        from torch.utils.data import TensorDataset
        cfg = dict(protocol(), predictor_epochs=1, fusion_epochs=1, batch_size=2,
                   hidden_dim=8, history_frames=3, residual_steps=2)
        normal = TensorDataset(torch.randn(4, 3, 4, 12), torch.randn(4, 4, 12))
        predictor = BackgroundPredictor(12, 8)
        history = fit_loop(predictor, normal, normal, cfg, "cpu", "predictor")
        self.assertEqual(len(history), 1)
        self.assertTrue(torch.isfinite(torch.tensor(history[0]["val"])))
        candidates = TensorDataset(torch.randn(4, 5, 4, 12), torch.randn(4, 4),
                                   torch.tensor([0., 1., 0., 1.]))
        candidates.counts = {0: 2, 1: 2}
        fusion = Fusion("learned_residual", 12, cfg, predictor)
        before = {k: v.clone() for k, v in predictor.state_dict().items()}
        fit_loop(fusion, candidates, candidates, cfg, "cpu", "fusion")
        self.assertTrue(all(torch.equal(before[k], v) for k, v in predictor.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
