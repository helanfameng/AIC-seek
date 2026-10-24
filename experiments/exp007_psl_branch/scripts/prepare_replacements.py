"""Download and inspect FIgLib replacement candidates without running a model.

All outputs are confined to a separate data directory. Original experiment six
files are read only. Candidate admission is a separate, explicit review record.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import threading
from urllib.parse import unquote, urljoin, urlparse

EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]
DATA = PROJECT / "data" / "exp007_replacement_candidates_20261023"
OLD_DATA = PROJECT / "data" / "exp006_background_residual"
OLD_SELECTION = PROJECT / "experiments" / "exp006_background_residual" / "data_selection.json"
LOCAL = threading.local()


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def session():
    if not hasattr(LOCAL, "session"):
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        item = requests.Session()
        item.headers["User-Agent"] = "AIC-seek-exp007-replacement-research/1.0"
        item.mount("https://", HTTPAdapter(max_retries=Retry(
            total=2, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))))
        LOCAL.session = item
    return LOCAL.session


def get(url):
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "cdn.hpwren.ucsd.edu":
        raise ValueError(f"Unexpected source host: {url}")
    response = session().get(url, timeout=(12, 40))
    response.raise_for_status()
    return response


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "a" and "href" in values:
            self.links.append(values["href"])


def candidate_pool():
    catalog = read_json(OLD_DATA / "catalog.json")
    selection = read_json(OLD_SELECTION)
    selected_ids = {s["id"] for s in selection["sequences"]}
    selected_dates = {s["event_group"] for s in selection["sequences"] if s["source"] == "figlib"}
    excluded = {s["id"] for s in catalog.get("excluded_sequences", [])}
    excluded |= {s["id"] for s in selection.get("quality_exclusions", [])}
    pool = [dict(s) for s in catalog["candidates"]
            if s["site"] in ("vo", "ml") and s["id"] not in selected_ids | excluded
            and s["event_group"] not in selected_dates]
    return sorted(pool, key=lambda s: (s["site"] != "vo", -int(s["date"]), s["id"]))


def inspect_one(candidate):
    candidate = dict(candidate)
    snap = DATA / "source_snapshots" / f"figlib_{candidate['id']}.html"
    if snap.exists():
        text = snap.read_text(encoding="utf-8")
    else:
        text = get(candidate["url"]).text
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_text(text, encoding="utf-8")
    parser = Links()
    parser.feed(text)
    frames = []
    for href in parser.links:
        filename = unquote(Path(urlparse(href).path).name)
        match = re.fullmatch(r"(\d{10})_([+-]\d+)\.jpg", filename, re.I)
        if not match:
            continue
        stamp, offset = map(int, match.groups())
        if -2400 <= offset <= 2400:
            frames.append({"timestamp": stamp, "offset_seconds": offset,
                           "url": urljoin(candidate["url"], href),
                           "file": f"images/{candidate['id']}/{filename}"})
    frames = sorted({f["timestamp"]: f for f in frames}.values(), key=lambda f: f["timestamp"])
    gaps = [b["timestamp"] - a["timestamp"] for a, b in zip(frames, frames[1:])]
    zeroes = sorted(f["timestamp"] - f["offset_seconds"] for f in frames)
    reasons = []
    if len(frames) < 60:
        reasons.append("fewer than 60 unique frames in +/-40 minutes")
    if not any(f["offset_seconds"] <= 0 for f in frames):
        reasons.append("no frame at or before official onset")
    if not any(f["offset_seconds"] >= 900 for f in frames):
        reasons.append("less than 15 minutes post-onset coverage")
    near = [f for f in frames if -600 <= f["offset_seconds"] <= 600]
    near_gaps = [b["timestamp"] - a["timestamp"] for a, b in zip(near, near[1:])]
    before = [f for f in frames if f["offset_seconds"] <= 0][-8:]
    history_ok = len(before) == 8 and all(
        b["timestamp"] - a["timestamp"] <= 120 for a, b in zip(before, before[1:]))
    if not history_ok:
        reasons.append("eight distinct causal history frames unavailable at official onset")
    candidate.update(source="figlib", proposed_split="diagnostic_test_v2", frames=frames,
                     official_onset=zeroes[len(zeroes) // 2] if zeroes else None,
                     metadata_eligible=not reasons, metadata_rejection_reasons=reasons,
                     frame_count=len(frames), max_gap_seconds=max(gaps, default=None),
                     onset_neighborhood_max_gap_seconds=max(near_gaps, default=None),
                     onset_history_available=history_ok,
                     onset_consistency_seconds=max(zeroes) - min(zeroes) if zeroes else None,
                     minimum_offset=min((f["offset_seconds"] for f in frames), default=None),
                     maximum_offset=max((f["offset_seconds"] for f in frames), default=None),
                     index_snapshot=str(snap.relative_to(DATA)).replace("\\", "/"),
                     index_snapshot_sha256=digest(snap), visual_review_status="not_reviewed")
    return candidate


def inspect():
    DATA.mkdir(parents=True, exist_ok=True)
    candidates = candidate_pool()
    write_json(DATA / "search_protocol.json", {
        "created_at": now(), "source": "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/",
        "sites": ["vo", "ml"], "ordering": "vo then ml; newest unused event first within site",
        "selection_basis": "source metadata and image review only; no detector or classifier outputs",
        "old_selection_sha256": digest(OLD_SELECTION), "old_catalog_sha256": digest(OLD_DATA / "catalog.json"),
        "candidate_ids": [s["id"] for s in candidates],
        "criteria": {"offset_window_seconds": [-2400, 2400], "minimum_unique_frames": 60,
                     "history_frames": 8, "maximum_history_gap_seconds": 120,
                     "visual": "daylight; identifiable natural smoke; no major foreground equipment occlusion"},
        "status": "candidate inspection only; not an active training/evaluation manifest"})
    rows = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(inspect_one, c): c for c in candidates}
        for future in as_completed(futures):
            c = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = dict(c, metadata_eligible=False, fetch_error=str(exc), frames=[])
            rows[c["id"]] = row
            print(json.dumps({k: row.get(k) for k in
                              ("id", "frame_count", "metadata_eligible", "metadata_rejection_reasons",
                               "max_gap_seconds", "onset_neighborhood_max_gap_seconds", "fetch_error")}), flush=True)
    write_json(DATA / "candidate_catalog.json", {"created_at": now(), "candidates": [rows[c["id"]] for c in candidates]})


def download_frame(frame):
    from PIL import Image
    path = DATA / frame["file"]
    if not path.exists():
        payload = get(frame["url"]).content
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    with Image.open(path) as picture:
        width, height = picture.size
        picture.verify()
    return dict(frame, bytes=path.stat().st_size, sha256=digest(path), width=width, height=height)


def download(ids, preview=False):
    candidates = read_json(DATA / "candidate_catalog.json")["candidates"]
    selected = [c for c in candidates if c["id"] in ids]
    if len(selected) != len(set(ids)):
        raise ValueError("Unknown or duplicated candidate IDs")
    if any(not c.get("metadata_eligible") for c in selected):
        raise ValueError("Candidate failed source metadata checks")
    records = {}
    tasks = []
    offsets = (-600, -300, 0, 60, 120, 180, 300, 600, 1200, 1800, 2400)
    for c in selected:
        frames = c["frames"]
        if preview:
            frames = sorted({min(frames, key=lambda f: abs(f["offset_seconds"] - t))["timestamp"]:
                             min(frames, key=lambda f: abs(f["offset_seconds"] - t)) for t in offsets}.values(),
                            key=lambda f: f["timestamp"])
        records[c["id"]] = dict(c, frames=[], download_scope="review_subset" if preview else "full_80_minute_window")
        tasks.extend((c["id"], f) for f in frames)
    completed = 0
    errors = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_frame, f): (seq_id, f) for seq_id, f in tasks}
        for future in as_completed(futures):
            seq_id, frame = futures[future]
            try:
                records[seq_id]["frames"].append(future.result())
                completed += 1
            except Exception as exc:
                errors.append({"id": seq_id, "file": frame["file"], "error": str(exc)})
            if (completed + len(errors)) % 10 == 0 or completed + len(errors) == len(tasks):
                print(f"downloaded {completed}/{len(tasks)}, errors {len(errors)}", flush=True)
    for c in records.values():
        c["frames"].sort(key=lambda f: f["timestamp"])
    filename = "preview_download.json" if preview else "download_manifest.json"
    write_json(DATA / filename, {"created_at": now(), "sequences": list(records.values()), "errors": errors,
                               "status": "downloaded candidate data; admission requires review record"})
    print(json.dumps({"manifest": str(DATA / filename), "downloaded": completed,
                      "errors": len(errors), "bytes": sum(f["bytes"] for c in records.values() for f in c["frames"])}), flush=True)
    if errors:
        raise RuntimeError(f"Download incomplete: {len(errors)} files")


def sheets(ids, full=False, roi=None):
    from PIL import Image, ImageDraw, ImageFont
    candidates = read_json(DATA / "candidate_catalog.json")["candidates"]
    out = DATA / "review"
    out.mkdir(parents=True, exist_ok=True)
    for c in candidates:
        if c["id"] not in ids:
            continue
        frames = [f for f in c["frames"] if (DATA / f["file"]).exists()]
        if not full:
            targets = (-600, 0, 60, 120, 180, 300, 600, 1200, 2400)
            frames = sorted({min(frames, key=lambda f: abs(f["offset_seconds"] - t))["timestamp"]:
                             min(frames, key=lambda f: abs(f["offset_seconds"] - t)) for t in targets}.values(),
                            key=lambda f: f["timestamp"])
        cell_width, cell_height = (640, 460) if not roi else (560, 380)
        columns = 3 if not full else 5
        page_size = 20 if full else len(frames)
        for start in range(0, len(frames), page_size):
            part = frames[start:start + page_size]
            rows = (len(part) + columns - 1) // columns
            canvas = Image.new("RGB", (columns * cell_width, 55 + rows * cell_height), "#141b23")
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 19)
            draw.text((12, 12), c["id"] + (" | ROI review" if roi else " | original frames"), fill="white", font=font)
            for i, frame in enumerate(part):
                with Image.open(DATA / frame["file"]) as original:
                    picture = original.convert("RGB")
                if roi:
                    w, h = picture.size
                    picture = picture.crop(tuple(round(v * (w if j % 2 == 0 else h)) for j, v in enumerate(roi)))
                picture.thumbnail((cell_width - 10, cell_height - 34))
                x, y = (i % columns) * cell_width, 55 + (i // columns) * cell_height
                canvas.paste(picture, (x + 5, y + 29))
                text = f"{frame['offset_seconds']/60:+.2f} min | {Path(frame['file']).name}"
                draw.text((x + 6, y + 3), text, fill="white", font=font)
            suffix = ("roi" if roi else "overview") + ("_full" if full else "")
            path = out / f"{c['id']}_{suffix}_{start//page_size+1:02d}.jpg"
            canvas.save(path, quality=91)
            print(str(path), flush=True)


def verify():
    from PIL import Image
    manifest = read_json(DATA / "download_manifest.json")
    old = read_json(OLD_SELECTION)
    frozen_sources = read_json(DATA / "search_protocol.json")
    assert digest(OLD_SELECTION) == frozen_sources["old_selection_sha256"], "Original selection changed"
    assert digest(OLD_DATA / "catalog.json") == frozen_sources["old_catalog_sha256"], "Original catalog changed"
    old_hashes = {f["sha256"] for c in old["sequences"] for f in c.get("frames", []) if "sha256" in f}
    old_dates = {c["event_group"] for c in old["sequences"] if c["source"] == "figlib"}
    forbidden_sites = {c["site"] for c in old["sequences"] if c["split"] in ("train", "val")}
    seen = set()
    results = []
    for c in manifest["sequences"]:
        assert c["site"] not in forbidden_sites, c["id"]
        assert c["event_group"] not in old_dates, c["id"]
        assert len(c["frames"]) == c["frame_count"], c["id"]
        stamps = [f["timestamp"] for f in c["frames"]]
        assert stamps == sorted(set(stamps)), c["id"]
        for f in c["frames"]:
            path = DATA / f["file"]
            assert path.stat().st_size == f["bytes"], str(path)
            with Image.open(path) as picture:
                assert list(picture.size) == [f["width"], f["height"]], str(path)
                picture.verify()
            actual = digest(path)
            assert actual == f["sha256"], str(path)
            assert actual not in old_hashes and actual not in seen, str(path)
            seen.add(actual)
        results.append({"id": c["id"], "frames": len(stamps), "bytes": sum(f["bytes"] for f in c["frames"]),
                        "offset_seconds": [c["minimum_offset"], c["maximum_offset"]],
                        "max_gap_seconds": c["max_gap_seconds"], "onset_history_available": c["onset_history_available"]})
    assert len({c["event_group"] for c in manifest["sequences"]}) == len(manifest["sequences"])
    selection_file = EXPERIMENT / "replacement_selection.json"
    if selection_file.exists():
        selected = read_json(selection_file)["selected_sequences"]
        assert {c["id"] for c in selected} == {c["id"] for c in manifest["sequences"]}
        actual_by_id = {c["id"]: c for c in manifest["sequences"]}
        for c in selected:
            actual = actual_by_id[c["id"]]
            assert c["frame_count"] == len(actual["frames"]), c["id"]
            assert c["max_gap_seconds"] == actual["max_gap_seconds"], c["id"]
            assert c["offset_seconds"] == [actual["minimum_offset"], actual["maximum_offset"]], c["id"]
    report = {"verified_at": now(), "sequences": results, "verified_frames": len(seen),
              "cross_split_site_overlap": False, "old_event_date_overlap": False,
              "old_manifest_hash_overlap": False, "new_bytes_rehashed": True,
              "jpeg_headers_and_dimensions_checked": True, "original_source_manifests_unchanged": True,
              "download_manifest_sha256": digest(DATA / "download_manifest.json"),
              "replacement_selection_sha256": digest(selection_file) if selection_file.exists() else None,
              "original_selection_sha256": digest(OLD_SELECTION)}
    write_json(DATA / "integrity_check.json", report)
    print(json.dumps(report), flush=True)


def summary():
    from PIL import Image, ImageDraw, ImageFont
    selection = read_json(EXPERIMENT / "replacement_selection.json")
    manifest = read_json(DATA / "download_manifest.json")
    by_id = {c["id"]: c for c in manifest["sequences"]}
    width, tile_height, row_header = 600, 350, 44
    chosen = selection["selected_sequences"]
    canvas = Image.new("RGB", (width * 3, len(chosen) * (tile_height + row_header)), "#141b23")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
    for row, reviewed in enumerate(chosen):
        c = by_id[reviewed["id"]]
        y = row * (tile_height + row_header)
        draw.text((12, y + 10), c["id"] + " | source-image region, no model output", fill="white", font=font)
        for column, offset in enumerate((-300, 540, 1200)):
            frame = min(c["frames"], key=lambda f: abs(f["offset_seconds"] - offset))
            with Image.open(DATA / frame["file"]) as picture:
                picture = picture.convert("RGB")
                w, h = picture.size
                roi = reviewed["review_region_xyxy_normalized"]
                picture = picture.crop(tuple(round(v * (w if j % 2 == 0 else h)) for j, v in enumerate(roi)))
                picture.thumbnail((width - 12, tile_height - 42))
                x = column * width
                canvas.paste(picture, (x + 6, y + row_header + 36))
            label = f"{frame['offset_seconds']/60:+.2f} min relative to official onset"
            draw.text((column * width + 8, y + row_header + 5), label, fill="#ffdb70", font=font)
    path = DATA / "review" / "replacement_comparison.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)
    print(str(path), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "preview", "download", "sheets", "verify", "summary"))
    parser.add_argument("--ids", nargs="+", default=[])
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--roi", nargs=4, type=float)
    args = parser.parse_args()
    if args.action in ("inspect", "preview", "download"):
        try:
            import truststore
            truststore.inject_into_ssl()
        except ImportError:
            pass
    if args.action == "inspect":
        inspect()
    elif args.action in ("preview", "download"):
        if not args.ids:
            parser.error("--ids required")
        download(args.ids, preview=args.action == "preview")
    elif args.action == "sheets":
        if not args.ids:
            parser.error("--ids required")
        sheets(args.ids, args.full, args.roi)
    elif args.action == "verify":
        verify()
    else:
        summary()


if __name__ == "__main__":
    main()
