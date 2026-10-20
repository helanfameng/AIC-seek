"""Checks for causality, padding interpretation and preservation of old results."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIRECTORY))
import replay_sequence as replay


class ReplayTests(unittest.TestCase):
    def test_all_81_decisions_use_only_the_recent_past(self):
        rows = list(range(81))
        decisions = list(replay.windows(rows, 20))
        self.assertEqual(len(decisions), 81)
        self.assertEqual(decisions[40][2], list(range(21, 41)))
        self.assertEqual(decisions[-1][2], list(range(61, 81)))
        for start, end, window in decisions:
            self.assertEqual(window[-1], end)
            self.assertLessEqual(len(window), 20)
            self.assertTrue(all(start <= index <= end for index in window))

    def test_padding_does_not_become_independent_observations_in_report(self):
        meta = [{"filename": "same.jpg", "offset_seconds": 0}] * 3
        tube = {"tube_id": 0, "start_frame": 0, "end_frame": 2, "logit": 1.0,
                "probability": None, "entries": [{"frame_idx": i, "confidence": 0.2,
                "bbox": [0.5, 0.5, 0.1, 0.1], "is_gap": False} for i in range(3)]}
        features = replay.tube_features(tube, meta, 1, None)
        self.assertEqual(features["span_frames"], 3)
        self.assertEqual(features["unique_observed_frames"], 1)

    def test_old_only_track_does_not_draw_a_crop_on_current_image(self):
        current = SimpleNamespace(detections=[])
        kept = [{"entries": [{"frame_idx": 0, "bbox": [0.5, 0.5, 0.1, 0.1], "is_gap": False}]}]
        effective = [SimpleNamespace(frame_id="old"), SimpleNamespace(frame_id="current")]
        def forbidden(*_):
            raise AssertionError("Old-only tube must not crop the current image.")
        shapes = replay.make_shapes(current, kept, effective, "current", 3072, 2048, {}, (forbidden, forbidden))
        self.assertEqual(shapes, [])

    def test_calibrator_decomposition_matches_existing_actual_predictions(self):
        model = Path("D:/AIC/pyronear_lab/models/v0.6.0/model.zip")
        old_result = ROOT / "experiments/exp002_detector_diagnostics/diagnostics.json"
        if not model.is_file() or not old_result.is_file():
            self.skipTest("Existing baseline unavailable.")
        with zipfile.ZipFile(model) as archive:
            calibrator = json.loads(archive.read("logistic_calibrator.json"))
        diagnostics = json.loads(old_result.read_text(encoding="utf-8-sig"))
        checked = 0
        for prefix in diagnostics["prefixes"]:
            effective = [{"filename": frame + ".jpg", "offset_seconds": int(frame.rsplit("_", 1)[1])}
                         for frame in prefix["effective_frame_ids"]]
            tubes = prefix["prediction"]["details"]["tubes"]["kept"]
            for tube in tubes:
                features = replay.tube_features(tube, effective, len(tubes), calibrator)
                self.assertAlmostEqual(features["recomputed_probability"], tube["probability"], places=12)
                checked += 1
        self.assertEqual(checked, 5)

    def test_preappearance_positives_are_kept_separate_from_postappearance(self):
        rows = [{"step_1based": i + 1, "source_frame_1based": i + 1,
                 "filename": str(i), "offset_seconds": offset, "max_probability": 0.8,
                 "current_candidate_count": 1, "kept_tube_count": 1, "is_positive": True}
                for i, offset in enumerate([-60, 0, 60])]
        result = replay.summarize(rows)
        self.assertEqual(result["first_positive_window"]["offset_seconds"], -60)
        self.assertEqual(result["first_postappearance_positive_window"]["offset_seconds"], 0)
        self.assertEqual(result["preappearance_positive_windows"], 1)
        self.assertEqual(result["positive_window_runs"][0]["positive_windows"], 3)

    def test_dry_run_checks_the_real_81_frame_inputs_without_creating_outputs(self):
        sequence = ROOT / "data/figlib_whittier_full_81"
        model = Path("D:/AIC/pyronear_lab/models/v0.6.0/model.zip")
        if not (sequence / "manifest.json").is_file() or not model.is_file():
            self.skipTest("Downloaded event or pinned package unavailable.")
        result = subprocess.run([sys.executable, "-B", str(SCRIPT_DIRECTORY / "replay_sequence.py"),
                                 "--sequence", str(sequence), "--model", str(model), "--dry-run"],
                                capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["selected_images"], 81)
        self.assertEqual(plan["offset_seconds"], [-2400, 2400])
        self.assertFalse(plan["uses_future_frames"])

    def test_existing_experiment_directory_is_rejected_before_loading_model(self):
        sequence = ROOT / "data/figlib_whittier_smoke_10"
        model = Path("D:/AIC/pyronear_lab/models/v0.6.0/model.zip")
        old_result = ROOT / "experiments/exp002_detector_diagnostics/diagnostics.json"
        if not model.is_file() or not old_result.is_file():
            self.skipTest("Existing baseline unavailable.")
        before = replay.sha256(old_result)
        result = subprocess.run([sys.executable, "-B", str(SCRIPT_DIRECTORY / "replay_sequence.py"),
                                 "--sequence", str(sequence), "--model", str(model),
                                 "--output-dir", str(old_result.parent)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Output directory already exists", result.stderr)
        self.assertEqual(replay.sha256(old_result), before)


if __name__ == "__main__":
    unittest.main()
