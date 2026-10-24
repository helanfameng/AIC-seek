"""Data-flow and control tests; no model weights or detector inference."""
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from support import ROOT, Inputs, Results, calibrated, empty_state, make_jobs, norm_to_xyxy, overlaps, select_steps
from runtime import Runtime, center_scale_matches
from stages import calibration_stage, crop_stage, features, oracle_stage, region_role


class Buffer:
    def __init__(self):
        self.rows = []

    def add(self, row):
        self.rows.append(row)


class FakeRuntime:
    def __init__(self, cfg, logit=0):
        self.model = SimpleNamespace(_cfg=cfg)
        self.logit, self.calls = logit, []

    def score(self, inputs, event, ids, entries, window, cfg, factor, force=False):
        self.calls.append((list(ids), [dict(e) for e in entries]))
        frame = inputs.frames[event][ids[-1]]
        cx, cy, w, h = window
        side = max(w * factor * frame["width"], h * factor * frame["height"])
        box = [max(0, round(cx * frame["width"] - side / 2)), max(0, round(cy * frame["height"] - side / 2)),
               min(frame["width"], round(cx * frame["width"] + side / 2)), min(frame["height"], round(cy * frame["height"] + side / 2))]
        return self.logit, box, "test_stub"


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = Inputs()

    def test_empty_region_semantics(self):
        self.assertEqual(empty_state(-60, 180, 1320), "before_official_onset")
        self.assertEqual(empty_state(0, 180, 1320), "not_discernible")
        self.assertEqual(empty_state(1380, 180, 1320), "visible_omitted_high_resolution_covered")
        self.assertEqual(empty_state(600, 180, 1320), "unresolved")
        self.assertEqual(empty_state(0, None, None), "unresolved")

    def test_shared_manual_images_not_four_copies(self):
        counts = {e: sum(bool(r["manual_box"]) for r in frames.values()) for e, frames in self.inputs.frames.items()}
        self.assertEqual(counts, {"alison": 17, "west": 20, "cranston": 6})
        self.assertEqual(len(make_jobs(self.inputs)), 12)
        self.assertEqual(len(make_jobs(self.inputs, "calibration")), 4)
        with self.assertRaises(ValueError):
            make_jobs(self.inputs, "crop", "west")

    def test_endpoint_filter_does_not_remove_past_window(self):
        job = make_jobs(self.inputs, "association", "west", "full_baseline")[0]
        steps = select_steps(self.inputs, job, 300, 300)
        self.assertEqual(len(steps), 1)
        offsets = [self.inputs.frames["west"][fid]["offset_seconds"] for fid in steps[0]["effective_frame_ids"]]
        self.assertLess(min(offsets), 300)
        self.assertEqual(max(offsets), 300)

    def test_crop_keeps_frames_T_C_N_and_mask_fixed(self):
        job = make_jobs(self.inputs, "crop", "alison", "full_baseline")[0]
        source = self.inputs.source("alison", "full_baseline")
        steps = select_steps(self.inputs, job, 180, 180)
        step = steps[0]
        tube = next(t for t in step["prediction"]["details"]["tubes"]["kept"]
                    if region_role(self.inputs, "alison", t, step["effective_frame_ids"])["current_manual_overlap"])
        runtime, output = FakeRuntime(source["run"]["effective_config"], tube["logit"]), Buffer()
        with contextlib.redirect_stdout(io.StringIO()):
            crop_stage(self.inputs, job, steps, runtime, output)
        self.assertEqual(len(output.rows), 5)
        frozen = {(r["frozen_T"], r["frozen_C"], r["frozen_N"], tuple(r["frame_ids"]), tuple(r["slot_is_gap"])) for r in output.rows}
        self.assertEqual(len(frozen), 1)
        self.assertTrue(all(call[1] == runtime.calls[0][1] for call in runtime.calls))
        variants = {r["variant"]: r for r in output.rows}
        self.assertEqual(variants["manual_extent_c1"]["contrast_reference"], "manual_extent_c1p5")
        self.assertEqual(variants["manual_extent_c1p5"]["contrast_reference"], "manual_center_original_extent")
        self.assertEqual(variants["manual_extent_c2"]["changed_component"], "context_only")
        self.assertTrue(all(r["manual_box_width_in_patch_unclipped"] > 0 for r in output.rows))

    def test_oracle_uses_only_visible_past_frames(self):
        job = make_jobs(self.inputs, "oracle", "alison")[0]
        cfg = self.inputs.source("alison", "full_baseline")["run"]["effective_config"]
        runtime, output = FakeRuntime(cfg), Buffer()
        with contextlib.redirect_stdout(io.StringIO()):
            oracle_stage(self.inputs, job, select_steps(self.inputs, job, 120, 120), runtime, output)
        for row in output.rows:
            offsets = [self.inputs.frames["alison"][fid]["offset_seconds"] for fid in row["frame_ids"]]
            self.assertEqual(offsets, [0, 60, 120])
            self.assertIsNone(row["probability"])

    def test_center_scale_gate_rejects_far_or_incompatible_boxes(self):
        def det(cx, width=.04):
            return SimpleNamespace(cx=cx, cy=.6, w=width, h=.04, class_id=0)
        def iou(a, b):
            return overlaps(norm_to_xyxy([a.cx, a.cy, a.w, a.h], 1, 1),
                            norm_to_xyxy([b.cx, b.cy, b.w, b.h], 1, 1))["iou"]
        result = center_scale_matches([det(.6)], [det(.615)], .2, iou, 3072, 2048, .75, 2.5)
        self.assertEqual(len(result), 1)
        self.assertEqual(center_scale_matches([det(.1)], [det(.6)], .2, iou, 3072, 2048, .75, 2.5), [])

    def test_original_association_reproduces_saved_tubes_and_west_gate(self):
        runtime = Runtime.__new__(Runtime)  # Only the association adapter, without loading a model.
        for setting in ("full_baseline", "high_resolution"):
            source = self.inputs.source("west", setting)
            cfg = source["run"]["effective_config"]
            for step in source["steps"]:
                runtime.rebuild(step, cfg, "original_iou020", 3072, 2048, self.inputs.protocol)
        step = next(s for s in self.inputs.source("west", "full_baseline")["steps"] if s["summary"]["offset_seconds"] == 300)
        cfg = self.inputs.source("west", "full_baseline")["run"]["effective_config"]
        counts = {}
        for variant in self.inputs.protocol["association_variants"]:
            kept, _, _ = runtime.rebuild(step, cfg, variant, 3072, 2048, self.inputs.protocol)
            counts[variant] = sum(region_role(self.inputs, "west", t, step["effective_frame_ids"])["current_manual_overlap"] for t in kept)
        self.assertEqual(counts, {"original_iou020": 0, "iou010": 1, "center_scale": 1})

    def test_fixed_features_reproduce_west_calibration_coupling(self):
        examples = {}
        for setting in ("full_baseline", "low_threshold"):
            job = make_jobs(self.inputs, "calibration", "west", setting)[0]
            output = Buffer()
            with contextlib.redirect_stdout(io.StringIO()):
                calibration_stage(self.inputs, job, select_steps(self.inputs, job, 360, 360), None, output)
            examples[setting] = next(r for r in output.rows if r["current_manual_overlap"] and r["variant"] == "original")
        self.assertEqual(examples["full_baseline"]["source_N"], 2)
        self.assertEqual(examples["low_threshold"]["source_N"], 4)
        self.assertTrue(examples["full_baseline"]["crosses_threshold"])
        self.assertFalse(examples["low_threshold"]["crosses_threshold"])
        self.assertAlmostEqual(examples["full_baseline"]["probability"], .8776281097472929, places=8)
        self.assertAlmostEqual(examples["low_threshold"]["probability"], .684741, places=5)

    def test_existing_job_directory_is_never_overwritten(self):
        scratch = (ROOT / ".cache/exp005_tests").resolve()
        scratch.mkdir(parents=True, exist_ok=True)
        self.assertTrue(scratch.is_relative_to(ROOT.resolve()))
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            target = Path(folder) / "job"
            target.mkdir()
            sentinel = target / "original.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                Results(target, {})
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
