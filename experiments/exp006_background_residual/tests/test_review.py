"""Batch-label preservation, unknown precedence and human-only proposal acceptance."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import EXPERIMENT
from review_assist import annotation, apply_batch, group_candidates, make_proposals, save_labels


def row(identity, timestamp=0, box=(.1, .1, .3, .3), source="figlib", split="train", sequence="s"):
    return {"id": identity, "timestamp": timestamp, "crop_box": list(box), "frame_file": f"{identity}.jpg",
            "source": source, "split": split, "sequence": sequence, "feature_file": None}


class ReviewTests(unittest.TestCase):
    def test_batch_preserves_old_smoke_background_and_uncertain(self):
        rows = [row(str(i)) for i in range(4)]
        state = {"candidates": {"0": {"label": 1, "matches_target": True},
                                "1": {"label": 0}, "2": {"label": None}}}
        old = copy.deepcopy(state["candidates"])
        result = apply_batch(state, rows, [{"id": r["id"], "label": 0} for r in rows])
        self.assertEqual(result, {"changed": 1, "preserved": 3})
        for key, value in old.items():
            self.assertEqual(state["candidates"][key], value)
        self.assertEqual(state["candidates"]["3"]["label"], 0)
        apply_batch(state, rows, [{"id": "0", "label": None}], overwrite=True)
        self.assertIsNone(state["candidates"]["0"]["label"])

    def test_invalid_batch_is_atomic_and_cannot_label_other_sequence(self):
        state = {"candidates": {}}
        with self.assertRaises(ValueError):
            apply_batch(state, [row("valid")], [{"id": "valid", "label": 1}, {"id": "other", "label": 0}])
        self.assertEqual(state, {"candidates": {}})
        with self.assertRaises(ValueError):
            apply_batch(state, [row("valid")], [{"id": "valid", "label": True}])

    def test_explicit_unknown_overrides_whole_clip_negative(self):
        state = {"normal_status": "confirmed_normal", "candidates": {"u": {"label": None}}}
        self.assertEqual(annotation(row("u", source="hpwren"), state), (True, None, False))
        self.assertEqual(annotation(row("background", source="hpwren"), state), (True, 0, False))
        state["human_visible_timestamp"] = 100
        self.assertEqual(annotation(row("before", timestamp=0), state), (False, None, False))

    def test_grouping_matches_geometry_not_local_tube_number(self):
        a, b = (.1, .1, .2, .2), (.7, .7, .8, .8)
        rows = [row("t0/tube0", 0, a), row("t0/tube1", 0, b),
                row("t1/tube1", 60, a), row("t1/tube0", 60, b), row("late", 400, a)]
        groups = group_candidates(rows, {})
        self.assertIn(["t0/tube0", "t1/tube1"], [g["ids"] for g in groups])
        self.assertIn(["t0/tube1", "t1/tube0"], [g["ids"] for g in groups])
        self.assertIn(["late"], [g["ids"] for g in groups])
        long = group_candidates([row(str(i), i * 60) for i in range(25)], {})
        self.assertEqual([len(g["ids"]) for g in long], [12, 12, 1])

    def test_proposals_use_train_labels_only_and_do_not_change_truth(self):
        import numpy as np
        output = EXPERIMENT / "outputs"
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            folder = Path(temporary)
            sequences = [{"id": "train", "source": "figlib", "split": "train"},
                         {"id": "val", "source": "figlib", "split": "val"},
                         {"id": "test", "source": "figlib", "split": "test"}]
            candidates = {"train": [row("negative", sequence="train"), row("positive", sequence="train")],
                          "val": [row("val_seed", split="val", sequence="val"), row("query", split="val", sequence="val")],
                          "test": [row("hidden", split="test", sequence="test")]}
            values = {"negative": [1., 0.], "positive": [0., 1.], "val_seed": [1., 0.], "query": [1., 0.], "hidden": [0., 1.]}
            for part in candidates.values():
                for value in part:
                    value["feature_file"] = value["id"] + ".npz"
                    value["official_probability"] = .999
                    np.savez(folder / value["feature_file"], features=np.tile(values[value["id"]], (8, 16, 1)))
            labels = {"sequences": {"train": {"candidates": {"negative": {"label": 0}, "positive": {"label": 1}}},
                                     "val": {"candidates": {"val_seed": {"label": 1}}}, "test": {"candidates": {}}}}
            old = copy.deepcopy(labels)
            result = make_proposals(folder, sequences, labels, candidates)
            self.assertEqual(result["seed_counts"], {"0": 1, "1": 1})
            self.assertEqual(result["candidates"]["query"]["label"], 0)
            self.assertNotIn("hidden", result["candidates"])
            self.assertEqual(old, labels)

    def test_save_is_durable_and_failed_write_keeps_previous_file(self):
        output = EXPERIMENT / "outputs"
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            path = Path(temporary) / "labels.json"
            save_labels(path, {"version": 1})
            with patch("review_assist.os.fsync", side_effect=OSError("write failed")):
                with self.assertRaises(OSError):
                    save_labels(path, {"version": 2})
            self.assertEqual(json.loads(path.read_text()), {"version": 1})
            self.assertEqual(list(Path(temporary).glob("*.tmp")), [])
            save_labels(path, {"version": 3})
            self.assertEqual(json.loads(path.read_text()), {"version": 3})

    def test_http_batch_acceptance_backup_and_stale_page_protection(self):
        import review
        from http.server import HTTPServer as RealHTTPServer
        from PIL import Image
        output = EXPERIMENT / "outputs"
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            base = Path(temporary)
            data, experiment = base / "data", base / "experiment"
            data.mkdir()
            folder = experiment / "outputs" / "fixture" / "baseline"
            folder.mkdir(parents=True)
            rows = [row(f"r{i}", i * 60) for i in range(4)]
            frames = [{"file": r["frame_file"], "timestamp": r["timestamp"], "offset_seconds": r["timestamp"]} for r in rows]
            manifest = {"sequences": [{"id": "s", "source": "figlib", "split": "train", "frames": frames}]}
            original = {"sequences": {"s": {"normal_status": "unreviewed", "frames": {},
                        "candidate_review_complete": {}, "human_visible_timestamp": None,
                        "candidates": {"r0": {"label": 0}}}}}
            save_labels(data / "labels.json", original)
            (folder / "run.json").write_text(json.dumps({"sequence_ids": ["s"]}))
            (folder / "s.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            (folder / "review_suggestions.json").write_text(json.dumps({"candidates": {
                r["id"]: {"label": None, "reason": "fixture"} for r in rows}}))
            for frame in frames:
                Image.new("RGB", (64, 64), "gray").save(data / frame["file"])
            ready, server_holder = threading.Event(), []

            def factory(address, handler):
                server = RealHTTPServer(("127.0.0.1", 0), handler)
                server_holder.append(server)
                ready.set()
                return server

            with patch.multiple(review, DATA=data, EXPERIMENT=experiment,
                                load_manifest=lambda: manifest, HTTPServer=factory):
                args = SimpleNamespace(run_id="fixture", setting="baseline", review_split="develop", port=0)
                worker = threading.Thread(target=review.serve, args=(args,), daemon=True)
                worker.start()
                self.assertTrue(ready.wait(5))
                server = server_holder[0]
                url = f"http://127.0.0.1:{server.server_port}"

                def get(path):
                    with urlopen(url + path, timeout=5) as response:
                        return json.load(response)

                def post(value):
                    request = Request(url + "/api/update", data=json.dumps(value).encode(),
                                      headers={"Content-Type": "application/json"})
                    with urlopen(request, timeout=5) as response:
                        return json.load(response)

                try:
                    with urlopen(url, timeout=5) as response:
                        self.assertIn("分组复核", response.read().decode())
                    with urlopen(url + "/thumbnail?sequence=s&id=r1", timeout=5) as response:
                        self.assertEqual(response.read()[:2], b"\xff\xd8")
                    before = get("/api/sequence?id=s")
                    value = post({"action": "batch", "sequence": "s", "version": before["version"],
                                  "items": [{"id": "r0", "label": 1}, {"id": "r1", "label": 1, "matches_target": True},
                                            {"id": "r2", "label": 0}]})
                    self.assertEqual((value["changed"], value["preserved"]), (2, 1))
                    with self.assertRaises(HTTPError) as conflict:
                        post({"action": "complete", "sequence": "s", "version": before["version"]})
                    self.assertEqual(conflict.exception.code, 409)
                    after = get("/api/sequence?id=s")
                    post({"action": "accept_suggestions", "sequence": "s", "version": after["version"],
                          "groups": [g["id"] for g in after["groups"]], "complete": True})
                    saved = json.loads((data / "labels.json").read_text())
                    self.assertEqual(saved["sequences"]["s"]["candidates"]["r0"]["label"], 0)
                    self.assertIsNone(saved["sequences"]["s"]["candidates"]["r3"]["label"])
                    self.assertTrue(saved["sequences"]["s"]["candidate_review_complete"]["baseline"])
                    backups = list(data.glob("labels.before_session_*.json"))
                    self.assertEqual(len(backups), 1)
                    self.assertEqual(json.loads(backups[0].read_text()), original)
                finally:
                    server.shutdown()
                    worker.join(5)
                    server.server_close()


if __name__ == "__main__":
    unittest.main()
