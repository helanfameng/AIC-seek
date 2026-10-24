"""Official public-source discovery, fixed site splits, resumable image downloads."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import random
import re
import shutil
import threading
from urllib.parse import unquote, urljoin, urlparse

from common import DATA, EXPERIMENT, check_split_integrity, digest, load_manifest, now, protocol, read_json, write_json

FIGLIB = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/"
CAMERAS = "https://hpwren.ucsd.edu/cameras/"
LOCAL = threading.local()


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "a" and "href" in values:
            self.links.append(values["href"])


def session():
    if not hasattr(LOCAL, "session"):
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        try:
            import truststore
            truststore.inject_into_ssl()
        except ImportError:
            pass
        item = requests.Session()
        item.headers["User-Agent"] = "AIC-seek-exp006-research/1.0"
        item.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.7,
                    status_forcelist=(429, 500, 502, 503, 504))))
        LOCAL.session = item
    return LOCAL.session


def get_text(url, snapshot=None):
    response = session().get(url, timeout=(15, 45))
    response.raise_for_status()
    text = response.text
    if snapshot:
        path = DATA / "source_snapshots" / snapshot
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return text


def links(text):
    parser = Links()
    parser.feed(text)
    return parser.links


def fire_identity(name):
    match = re.match(r"(?P<date>\d{8})[._-].*?(?P<camera>[a-z0-9]+-[nsew]-mobo-c)$", name, re.I)
    if not match:
        return None
    camera = match["camera"].lower()
    date = match["date"]
    label = name[len(date):name.lower().rfind(camera)].strip("_- .").lower()
    # One selection per date is conservative about multi-camera views of the same fire.
    return {"id": name, "camera": camera, "site": camera.split("-")[0], "date": date,
            "label": label, "event_group": date}


def discover():
    cfg = protocol()
    text = get_text(FIGLIB + "index.html", "figlib_index.html")
    candidates = []
    for href in links(text):
        name = href.removesuffix("/index.html").strip("/")
        row = fire_identity(name)
        if not row:
            continue
        if row["site"] in cfg["excluded_development_sites"]:
            continue
        if any(word in name.lower() for word in ("night", "drill", "vegm", "prescrib", "struct", "house",
                                                   "jeep", "ship", "bonhomme", "backfire", "cardboard")):
            continue
        if any(word.lower() in name.lower() for word in cfg["excluded_development_events"]):
            continue
        row["url"] = urljoin(FIGLIB, href)
        candidates.append(row)
    write_json(DATA / "catalog.json", {"created_at": now(), "source": FIGLIB, "candidates": candidates})
    print(json.dumps({"eligible_sequences": len(candidates),
                      "by_site": dict(Counter(x["site"] for x in candidates))}, indent=2))
    camera_html = get_text(CAMERAS, "cameras.html")
    for number, line in enumerate(camera_html.splitlines()):
        if any(term in line for term in ("fetch(", "cdn.hpwren", ".json", ".php", "list-objects", "getImageList")):
            if not ("googleapis" in line or "apiKey" in line):
                print(f"CAMERA SOURCE {number}: {line[:350]}")


def select_fires():
    manifest_path = DATA / "manifest.json"
    if manifest_path.exists() and any("sha256" in f for s in read_json(manifest_path)["sequences"] for f in s["frames"]):
        raise RuntimeError("Images already downloaded; selection is frozen. Use Data to resume instead of reselecting.")
    cfg = protocol()
    catalog = read_json(DATA / "catalog.json")
    invalid = {x["id"] for x in catalog.get("excluded_sequences", [])}
    candidates = [x for x in catalog["candidates"] if x["id"] not in invalid]
    sites = sorted(set(x["site"] for x in candidates))
    rng = random.Random(cfg["seed"])
    best = None
    # Deterministic search uses metadata only, never any model scores.
    for _ in range(1500):
        shuffled = sites[:]
        rng.shuffle(shuffled)
        n = len(sites)
        by_split = {"test": set(shuffled[:max(5, n // 4)]),
                    "val": set(shuffled[max(5, n // 4):max(5, n // 4) + max(4, n // 4)])}
        by_split["train"] = set(sites) - by_split["val"] - by_split["test"]
        selected, used_dates, used_names = [], set(), set()
        valid = True
        for split in ("test", "val", "train"):
            pool = [x for x in candidates if x["site"] in by_split[split]]
            rng.shuffle(pool)
            counts = Counter()
            for _slot in range(cfg["fire_target"][split]):
                available = [x for x in pool if x["date"] not in used_dates and
                             (x["label"] in ("fire", "") or x["label"] + x["date"][:4] not in used_names)]
                if not available:
                    valid = False
                    break
                available.sort(key=lambda x: (counts[x["site"]], -int(x["date"])))
                row = dict(available[0], split=split, source="figlib", review_status="pending",
                           pretraining_overlap="unknown")
                selected.append(row)
                used_dates.add(row["date"])
                if row["label"] not in ("fire", ""):
                    used_names.add(row["label"] + row["date"][:4])
                counts[row["site"]] += 1
            if not valid:
                break
        if valid:
            score = len(set(x["site"] for x in selected)) * 100 + sum(int(x["date"][:4]) for x in selected)
            if best is None or score > best[0]:
                best = (score, selected)
    if best is None:
        raise RuntimeError("Not enough distinct events under the requested site split")
    selected = best[1]
    for seq in selected:
        snapshot_name = "figlib_" + seq["id"] + ".html"
        snapshot_path = DATA / "source_snapshots" / snapshot_name
        text = snapshot_path.read_text(encoding="utf-8") if snapshot_path.exists() else get_text(seq["url"], snapshot_name)
        frames = []
        for href in links(text):
            name = unquote(Path(urlparse(href).path).name)
            match = re.fullmatch(r"(\d{10})_([+-]\d+)\.jpg", name, re.I)
            if not match:
                continue
            stamp, offset = map(int, match.groups())
            if -2400 <= offset <= 2400:
                frames.append({"timestamp": stamp, "offset_seconds": offset,
                    "url": urljoin(seq["url"], href),
                    "file": f"images/figlib/{seq['id']}/{name}"})
        frames = sorted({f["timestamp"]: f for f in frames}.values(), key=lambda x: x["timestamp"])
        if len(frames) < 60 or not any(f["offset_seconds"] <= 0 for f in frames) or not any(f["offset_seconds"] >= 900 for f in frames):
            catalog.setdefault("excluded_sequences", []).append({"id": seq["id"],
                "reason": f"Does not meet cadence/coverage requirement: {len(frames)} unique frames"})
            write_json(DATA / "catalog.json", catalog)
            print("Excluded by cadence/coverage:", seq["id"], len(frames), flush=True)
            return select_fires()
        zeroes = [f["timestamp"] - f["offset_seconds"] for f in frames]
        seq["official_onset"] = sorted(zeroes)[len(zeroes) // 2]
        seq["frames"] = frames
        print(seq["split"], seq["id"], len(frames), flush=True)
    check_split_integrity(selected)
    write_json(EXPERIMENT / "data_selection.json", {"created_at": now(), "seed": cfg["seed"],
        "selection_rule": "metadata only; distinct fire dates; site-disjoint splits; no development sites",
        "pretraining_overlap": "not verified", "sequences": selected})
    write_json(DATA / "manifest.json", {"created_at": now(), "sequences": selected})


def download_one(frame):
    from PIL import Image
    frame = dict(frame)
    frame.pop("download_error", None)
    target = DATA / frame["file"]
    try:
        if target.is_file() and target.stat().st_size > 1000:
            with Image.open(target) as image:
                image.verify()
            with Image.open(target) as image:
                width, height = image.size
            return dict(frame, bytes=target.stat().st_size, sha256=digest(target), width=width, height=height)
        if shutil.disk_usage(DATA).free < 2 * 1024 ** 3:
            raise RuntimeError("Less than 2 GiB free; download paused without deleting files")
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(target.suffix + ".part")
        with session().get(frame["url"], stream=True, timeout=(15, 90)) as response:
            response.raise_for_status()
            with part.open("wb") as out:
                for chunk in response.iter_content(256 * 1024):
                    out.write(chunk)
        with Image.open(part) as image:
            image.verify()
        with Image.open(part) as image:
            width, height = image.size
        part.replace(target)
        return dict(frame, bytes=target.stat().st_size, sha256=digest(target), width=width, height=height)
    except Exception as error:
        return dict(frame, download_error=f"{type(error).__name__}: {error}")


def select_normal():
    cfg = protocol()
    manifest = read_json(DATA / "manifest.json")
    if any(s["source"] == "hpwren" for s in manifest["sequences"]):
        if any((DATA / f["file"]).exists() for s in manifest["sequences"] if s["source"] == "hpwren" for f in s["frames"]):
            raise RuntimeError("Normal images already downloaded; keep the frozen selection and resume download")
        manifest["sequences"] = [s for s in manifest["sequences"] if s["source"] == "figlib"]
    catalog = read_json(DATA / "catalog.json")["candidates"]
    site_splits = check_split_integrity(manifest["sequences"])
    newest = datetime.now(timezone.utc).date() - timedelta(days=2)
    # Q4/Q5/Q6 cover Pacific 09..12 / 12..15 / 15..18.
    dates = [(newest - timedelta(days=d)).strftime("%Y%m%d") for d in (0, 3, 7, 11, 16)]
    output = []
    rng = random.Random(cfg["seed"])
    for split in ("train", "val", "test"):
        target = int(cfg["normal_hours_target"][split] / cfg["normal_clip_hours"])
        sites = sorted(s for s, part in site_splits.items() if part == split)
        rng.shuffle(sites)
        for site in sites:
            slot = len([x for x in output if x["split"] == split])
            if slot >= target:
                break
            cameras = sorted(set(x["camera"] for x in catalog if x["site"] == site))
            preferred = {s["camera"] for s in manifest["sequences"] if s["site"] == site}
            cameras.sort(key=lambda c: (c not in preferred, c))
            date_order = dates[slot % len(dates):] + dates[:slot % len(dates)]
            quarter = (4, 5, 6)[slot % 3]
            found_clip = False
            for date in date_order:
              if found_clip:
                break
              for camera in cameras:
                list_url = f"https://cdn.hpwren.ucsd.edu/hpwren-cameras/{camera}/{date[:4]}/{date}/{date}_{camera}_Q{quarter}.txt"
                try:
                    text = get_text(list_url)
                except Exception:
                    continue
                all_stamps = sorted(set(int(m) for m in re.findall(r"(?m)^(\d{10})", text)))
                if not all_stamps:
                    continue
                start = all_stamps[0] // 3600 * 3600
                end = start + int(cfg["normal_clip_hours"] * 3600)
                bins = {}
                for stamp in all_stamps:
                    if start <= stamp < end:
                        bins.setdefault((stamp - start) // cfg["sampling_seconds"], stamp)
                stamps = [bins[k] for k in sorted(bins)]
                if len(stamps) < 110 or any(b - a > cfg["maximum_history_gap_seconds"] for a, b in zip(stamps, stamps[1:])):
                    continue
                seq_id = f"normal_{date}_Q{quarter}_{camera}"
                seq = {"id": seq_id, "source": "hpwren", "split": split, "site": site,
                       "camera": camera, "review_status": "pending_normal_review",
                       "start_utc": start, "end_utc_exclusive": end,
                       "requested_hours": cfg["normal_clip_hours"], "list_url": list_url,
                       "archive_timezone": "America/Los_Angeles",
                       "selection_reason": f"fixed color camera; diverse dates; Pacific-local daylight Q{quarter}; continuous listing"}
                seq["frames"] = [{"timestamp": stamp, "offset_seconds": stamp - start,
                    "url": f"https://cdn.hpwren.ucsd.edu/MTA/{camera}/large/{date}/Q{quarter}/{stamp}.jpg",
                    "file": f"images/hpwren/{seq_id}/{stamp}.jpg"} for stamp in stamps]
                snapshot = DATA / "source_snapshots" / f"{seq_id}.txt"
                snapshot.write_text(text, encoding="utf-8")
                output.append(seq)
                print(split, seq_id, len(stamps), "pending normal review", flush=True)
                found_clip = True
                break
        found = len([x for x in output if x["split"] == split])
        if found < target:
            raise RuntimeError(f"Only {found}/{target} normal clips found in split {split}; existing manifest unchanged")
    manifest["sequences"].extend(output)
    check_split_integrity(manifest["sequences"])
    manifest["updated_at"] = now()
    write_json(DATA / "manifest.json", manifest)
    selection = read_json(EXPERIMENT / "data_selection.json")
    selection["sequences"] = manifest["sequences"]
    write_json(EXPERIMENT / "data_selection.json", selection)


def replace_night_sequences():
    """Keep frozen site splits; replace visibly nocturnal sequences before training."""
    from PIL import Image, ImageStat
    from dateutil.tz import gettz
    manifest = read_json(DATA / "manifest.json")
    site_splits = check_split_integrity(manifest["sequences"])
    catalog = read_json(DATA / "catalog.json")
    excluded = {x["id"] for x in catalog.get("excluded_sequences", [])}
    zone = gettz("America/Los_Angeles")
    rejected = []
    for seq in manifest["sequences"]:
        if seq["source"] != "figlib":
            continue
        onset = min(seq["frames"], key=lambda f: abs(f["offset_seconds"]))
        with Image.open(DATA / onset["file"]) as image:
            image.thumbnail((160, 120))
            luminance = ImageStat.Stat(image.convert("L")).mean[0]
        local_hour = datetime.fromtimestamp(seq["official_onset"], timezone.utc).astimezone(zone).hour
        if luminance < 35 or local_hour < 5 or local_hour >= 21:
            rejected.append(seq)
    if not rejected:
        print("No nocturnal sequences require replacement")
        return
    rejected_ids = {s["id"] for s in rejected}
    kept = [s for s in manifest["sequences"] if s["id"] not in rejected_ids]
    used_dates = {s["date"] for s in kept if s["source"] == "figlib"}
    used_names = {s["label"] + s["date"][:4] for s in kept if s["source"] == "figlib" and s["label"] != "fire"}
    for old in rejected:
        pool = [x for x in catalog["candidates"] if x["id"] not in excluded | rejected_ids and
                site_splits.get(x["site"]) == old["split"] and x["date"] not in used_dates and
                (x["label"] == "fire" or x["label"] + x["date"][:4] not in used_names)]
        pool.sort(key=lambda x: (x["site"] != old["site"], -int(x["date"])))
        replacement = None
        for candidate in pool:
            snapshot_name = "figlib_" + candidate["id"] + ".html"
            snapshot = DATA / "source_snapshots" / snapshot_name
            text = snapshot.read_text(encoding="utf-8") if snapshot.exists() else get_text(candidate["url"], snapshot_name)
            frames = {}
            for href in links(text):
                name = unquote(Path(urlparse(href).path).name)
                match = re.fullmatch(r"(\d{10})_([+-]\d+)\.jpg", name, re.I)
                if not match:
                    continue
                stamp, offset = map(int, match.groups())
                if -2400 <= offset <= 2400:
                    frames[stamp] = {"timestamp": stamp, "offset_seconds": offset, "url": urljoin(candidate["url"], href),
                                    "file": f"images/figlib/{candidate['id']}/{name}"}
            frames = sorted(frames.values(), key=lambda f: f["timestamp"])
            if len(frames) < 60:
                continue
            onset = sorted(f["timestamp"] - f["offset_seconds"] for f in frames)[len(frames) // 2]
            hour = datetime.fromtimestamp(onset, timezone.utc).astimezone(zone).hour
            if not 8 <= hour < 17:
                continue
            probe = download_one(min(frames, key=lambda f: abs(f["offset_seconds"])))
            if "download_error" in probe:
                continue
            with Image.open(DATA / probe["file"]) as image:
                image.thumbnail((160, 120))
                if ImageStat.Stat(image.convert("L")).mean[0] < 35:
                    continue
            replacement = dict(candidate, source="figlib", split=old["split"], frames=frames,
                               official_onset=onset, review_status="pending", pretraining_overlap="unknown",
                               selection_reason="replacement of a nocturnal sequence; fixed station split; daylight onset")
            break
        if replacement is None:
            raise RuntimeError(f"No daylight replacement found for {old['id']}; original manifest preserved")
        kept.append(replacement)
        used_dates.add(replacement["date"])
        used_names.add(replacement["label"] + replacement["date"][:4])
        print("Daylight replacement:", old["id"], "->", replacement["id"], flush=True)
    check_split_integrity(kept)
    manifest["sequences"] = kept
    manifest.setdefault("quality_exclusions", []).extend(
        {"id": s["id"], "split": s["split"], "reason": "nocturnal onset; outside the daylight RGB scope"} for s in rejected)
    manifest["download_status"] = "partial"
    write_json(DATA / "manifest.json", manifest)
    selection = read_json(EXPERIMENT / "data_selection.json")
    selection["sequences"] = [{k: v for k, v in s.items()} for s in kept]
    selection["quality_exclusions"] = manifest["quality_exclusions"]
    selection["selection_rule"] += "; nocturnal onsets replaced within the original station split before training"
    write_json(EXPERIMENT / "data_selection.json", selection)
    print("Excluded night images retained on disk for traceability; not part of the active manifest")


def download(source="all", workers=4):
    manifest = load_manifest()
    jobs = [(seq, i, frame) for seq in manifest["sequences"]
            if source == "all" or seq["source"] == source
            for i, frame in enumerate(seq["frames"])]
    failures = 0
    with ThreadPoolExecutor(workers) as executor:
        futures = {executor.submit(download_one, frame): (seq, i) for seq, i, frame in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            seq, index = futures[future]
            result = future.result()
            failures += "download_error" in result
            seq["frames"][index] = result
            if done % 50 == 0 or done == len(jobs):
                manifest["updated_at"] = now()
                write_json(DATA / "manifest.json", manifest)
                print(f"Downloaded/verified {done}/{len(jobs)}; failed={failures}", flush=True)
    if failures:
        raise RuntimeError(f"{failures} downloads failed; rerun the same command to resume")
    downloaded = [f for s in manifest["sequences"] for f in s["frames"] if "sha256" in f]
    expected = sum(len(s["frames"]) for s in manifest["sequences"])
    manifest["download_status"] = "complete" if len(downloaded) == expected else "partial"
    manifest["downloaded_images"] = len(downloaded)
    manifest["expected_images"] = expected
    manifest["downloaded_bytes"] = sum(f["bytes"] for f in downloaded)
    write_json(DATA / "manifest.json", manifest)
    source = ["# 实验6数据来源与下载状态", "",
        f"更新：{now()}。状态：{manifest['download_status']}；{len(downloaded)}/{expected}张原图，"
        f"{manifest['downloaded_bytes'] / 1024**3:.3f} GiB。", "",
        "- FIgLib：https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/index.html",
        "- HPWREN相机与历史原图：https://hpwren.ucsd.edu/cameras/",
        "- 原图保持下载字节，不缩放、不重编码。每张原图的URL、时间戳、尺寸、字节数和SHA256在manifest.json。",
        "- 遵循来源要求，引用HPWREN / University of California San Diego。",
        "- 正常序列只是候选；人工确认前不作为无烟训练/测试真值。", "",
        "| 划分 | 火情序列 | 正常候选片段 | 图片 | 正常候选计划时长 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for split in ("train", "val", "test"):
        seqs = [s for s in manifest["sequences"] if s["split"] == split]
        source.append(f"| {split} | {sum(s['source']=='figlib' for s in seqs)} | "
                      f"{sum(s['source']=='hpwren' for s in seqs)} | {sum(len(s['frames']) for s in seqs)} | "
                      f"{sum(s.get('requested_hours',0) for s in seqs)}小时 |")
    source.extend(["", "数据按来源站点前缀隔离；排除sp/lp/hp/syp开发站点；选样使用来源元数据，未使用模型成绩。",
        "同日火情保守合并、同年同名火情去重；跨日同一火情的真实身份仍应在人工复核时检查。",
        "初次下载后发现的夜间序列已从活动清单移除；原图另存excluded_night目录追溯，不参加本轮训练/验证/测试。",
        "训练/验证/测试与官方预训练数据的重叠尚未核实。",
        "图像索引快照在source_snapshots，固定选样清单在实验6目录的data_selection.json。", ""])
    (DATA / "SOURCE.md").write_text("\n".join(source), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["discover", "select", "normal", "quality", "download"])
    parser.add_argument("--source", choices=["all", "figlib", "hpwren"], default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.action == "discover":
        discover()
    elif args.action == "select":
        select_fires()
    elif args.action == "normal":
        select_normal()
    elif args.action == "quality":
        replace_night_sequences()
    else:
        download(args.source, args.workers)


if __name__ == "__main__":
    main()
