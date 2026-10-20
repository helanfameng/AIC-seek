"""Checks for acquisition mapping, experimental isolation and output protection."""
import copy
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "scripts"))
import download_events as data
import run_experiments as runner


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = data.load_protocol()

    def test_official_snapshots_and_frame_mapping(self):
        all_hashes = set()
        for event in self.config["events"]:
            folder = data.event_directory(self.config, event)
            names = data.parse_index((folder / "source_index.html").read_bytes(), event)
            manifest = data.read_json(folder / "manifest.json")
            self.assertEqual([row["filename"] for row in manifest["files"]], names)
            self.assertEqual(len(names), 81)
            self.assertEqual(names[40], f"{event['origin_timestamp']}_+00000.jpg")
            offsets = [row["offset_seconds"] for row in manifest["files"]]
            self.assertEqual(offsets, list(range(-2400, 2401, 60)))
            hashes = {row["sha256"] for row in manifest["files"]}
            self.assertFalse(all_hashes.intersection(hashes))
            all_hashes.update(hashes)
        old = data.read_json(data.ROOT / "data/figlib_whittier_full_81/manifest.json")
        self.assertFalse(all_hashes.intersection(row["sha256"] for row in old["files"]))

    def test_index_rejects_a_missing_or_extra_frame(self):
        event = self.config["events"][0]
        links = [f'<a href="{name}">{name}</a>' for name in data.expected_names(event)]
        self.assertEqual(len(data.parse_index("".join(links).encode(), event)), 81)
        with self.assertRaises(ValueError):
            data.parse_index("".join(links[:-1]).encode(), event)
        with self.assertRaises(ValueError):
            data.parse_index(("".join(links) + '<a href="1527972450_+02460.jpg">extra</a>').encode(), event)
        parser = data.DirectoryLinks(data.CDN + event["event_id"] + "/index.html")
        parser.feed('<a href="https://example.com/1527969990_+00000.jpg">external</a>'
                    '<a href="../1527969990_+00000.jpg">parent</a>')
        self.assertEqual(parser.names, set())

    def test_hash_corruption_is_rejected_before_decode(self):
        event = self.config["events"][0]
        image = data.event_directory(self.config, event) / data.expected_names(event)[0]
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            data.inspect_image(image, "0" * 64)

    def test_twelve_complete_replays_only_change_two_parameters(self):
        jobs = runner.build_jobs(self.config, runner.DEFAULT_MODEL, "cuda", "unit_test")
        self.assertEqual(len(jobs), 12)
        self.assertEqual(len({job["output"] for job in jobs}), 12)
        expected = {item["key"]: (item["confidence"], item["image_size"]) for item in self.config["settings"]}
        for job in jobs:
            command = job["command"]
            option = lambda key: command[command.index(key) + 1]
            self.assertEqual((float(option("--confidence")), int(option("--image-size"))), expected[job["setting"]])
            self.assertEqual(option("--window"), "20")
            self.assertEqual(option("--detection-mode"), "per-frame-cache")
            self.assertEqual(option("--warmup"), "1")
            self.assertEqual(option("--expected-model-sha256"), self.config["model_sha256"])
            event = next(item for item in self.config["events"] if item["key"] == job["event"])
            self.assertEqual(Path(option("--sequence")), data.event_directory(self.config, event))
            self.assertNotIn("--limit", command)
            self.assertNotIn("--offset-min", command)
            self.assertNotIn("--offset-max", command)
            self.assertEqual(job["images"], 81)

    def test_all_windows_use_only_current_and_past_images(self):
        api = runpy.run_path(str(runner.replay_path(self.config)))
        for event in self.config["events"]:
            rows = [{"timestamp": event["origin_timestamp"] + offset} for offset in range(-2400, 2401, 60)]
            windows = list(api["windows"](rows, 20))
            self.assertEqual(len(windows), 81)
            for current, (start, end, window) in enumerate(windows):
                self.assertEqual(end, current)
                self.assertEqual(len(window), min(current + 1, 20))
                self.assertTrue(all(row["timestamp"] <= rows[current]["timestamp"] for row in window))
                self.assertEqual(window[-1], rows[current])

    def test_existing_output_aborts_before_data_check_or_inference(self):
        jobs = [{"output": str(EXPERIMENT / "does_not_exist")}, {"output": str(EXPERIMENT)}]
        with patch.object(sys, "argv", ["run_experiments.py", "run", "--run-id", "unit_test"]), \
                patch.object(runner, "build_jobs", return_value=jobs), \
                patch.object(runner, "offline_check") as offline, \
                patch.object(runner, "check_runtime") as runtime:
            with self.assertRaises(FileExistsError):
                runner.main()
            offline.assert_not_called()
            runtime.assert_not_called()

    def test_invalid_run_ids_and_changed_replay_are_rejected(self):
        for run_id in ["../other", "D:\\outside", "", "/tmp", "x" * 65]:
            with self.assertRaises(ValueError):
                runner.run_directory(run_id)
        changed = copy.deepcopy(self.config)
        changed["replay_sha256_lf"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "replay changed"):
            runner.replay_path(changed)


if __name__ == "__main__":
    unittest.main()
