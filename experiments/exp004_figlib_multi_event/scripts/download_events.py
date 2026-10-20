"""Download or verify the three frozen FIgLib event sequences for exp004."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from urllib.parse import quote, unquote, urljoin, urlsplit
from urllib.request import Request, urlopen
import uuid


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
CONFIG = EXPERIMENT / "events.json"
CDN = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/"
NAME = re.compile(r"^(\d{10})_([+-]\d{5})\.jpg$")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol():
    config = read_json(CONFIG)
    events = config["events"]
    for field in ["key", "name", "date", "camera", "origin_timestamp"]:
        if len({event[field] for event in events}) != 3:
            raise ValueError(f"Expected three distinct event {field} values.")
    if len(events) != 3 or config["images_per_event"] != 81:
        raise ValueError("This protocol requires three complete 81-image events.")
    if (config["offset_min_seconds"], config["offset_max_seconds"], config["interval_seconds"]) != (-2400, 2400, 60):
        raise ValueError("Unexpected sampling protocol.")
    settings = config["settings"]
    if len(settings) != 4 or len({item["key"] for item in settings}) != 4 or {(item["confidence"], item["image_size"]) for item in settings} != {(0.1, 1024), (0.05, 1024), (0.1, 1536), (0.05, 1536)}:
        raise ValueError("The frozen two-by-two parameter design changed.")
    data_root = (ROOT / config["dataset_root_relative"]).resolve()
    if not data_root.is_relative_to((ROOT / "data").resolve()):
        raise ValueError("Downloads must remain under the project data directory.")
    return config


def select_events(config, selector):
    return config["events"] if selector == "all" else [next(event for event in config["events"] if event["key"] == selector)]


def event_directory(config, event):
    path = (ROOT / config["dataset_root_relative"] / event["event_id"]).resolve()
    if not path.is_relative_to((ROOT / "data").resolve()):
        raise ValueError("Event path escaped the project data directory.")
    return path


def expected_names(event):
    origin = event["origin_timestamp"]
    return [f"{origin + offset}_{offset:+06d}.jpg" for offset in range(-2400, 2401, 60)]


class DirectoryLinks(HTMLParser):
    def __init__(self, index_url):
        super().__init__()
        self.index_url = index_url
        self.base_path = urlsplit(index_url).path.rsplit("/", 1)[0] + "/"
        self.names = set()

    def handle_starttag(self, tag, attributes):
        if tag.lower() != "a":
            return
        target = urlsplit(urljoin(self.index_url, dict(attributes).get("href", "")))
        name = unquote(target.path.rsplit("/", 1)[-1])
        if (target.scheme == "https" and target.netloc == "cdn.hpwren.ucsd.edu"
                and unquote(target.path) == self.base_path + name and NAME.fullmatch(name)):
            self.names.add(name)


def parse_index(data, event):
    parser = DirectoryLinks(CDN + event["event_id"] + "/index.html")
    parser.feed(data.decode("utf-8"))
    wanted = expected_names(event)
    if sorted(parser.names) != wanted:
        raise ValueError(f"Official index is not the complete frozen sequence: {event['event_id']}")
    return wanted


def fetch_bytes(url):
    target = urlsplit(url)
    if target.scheme != "https" or target.netloc != "cdn.hpwren.ucsd.edu":
        raise ValueError("Downloads must use the official HTTPS CDN.")
    # Windows curl uses the system TLS stack, which also works in environments
    # where the Python/OpenSSL connection to this CDN is interrupted.
    curl = shutil.which("curl.exe") if os.name == "nt" else None
    if curl:
        response = subprocess.run(
            [curl, "--fail", "--silent", "--show-error", "--proto", "=https",
             "--connect-timeout", "15", "--max-time", "60", "--retry", "2", "--url", url],
            capture_output=True, check=True)
        return response.stdout
    request = Request(url, headers={"User-Agent": "AIC-seek/exp004 (FIgLib research)"})
    with urlopen(request, timeout=40) as response:
        final = urlsplit(response.url)
        if final.scheme != "https" or final.netloc != "cdn.hpwren.ucsd.edu":
            raise ValueError("Unexpected download redirect.")
        return response.read()


def inspect_image(path, expected_hash=None):
    from PIL import Image
    digest = sha256(path)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"Acquisition hash mismatch: {path}")
    with Image.open(path) as image:
        if image.format != "JPEG" or image.mode != "RGB":
            raise ValueError(f"Expected an original color JPEG: {path}")
        width, height = image.size
        image.verify()
    with Image.open(path) as image:
        image.load()
    return {"bytes": path.stat().st_size, "sha256": digest, "width": width, "height": height}


def verify_event(config, event):
    folder = event_directory(config, event)
    manifest = read_json(folder / "manifest.json")
    wanted = expected_names(event)
    if manifest["event"] != event["event_id"] or manifest["count"] != 81:
        raise ValueError(f"Wrong event manifest: {folder}")
    if sorted(path.name for path in folder.glob("*.jpg")) != wanted:
        raise ValueError(f"Missing or extra JPEGs: {folder}")
    records = manifest["files"]
    if [row["filename"] for row in records] != wanted:
        raise ValueError(f"Manifest frame mapping changed: {folder}")
    sizes = set()
    total_bytes = 0
    for record in records:
        name = record["filename"]
        timestamp, offset = map(int, NAME.fullmatch(name).groups())
        if record["timestamp"] != timestamp or record["offset_seconds"] != offset or timestamp - offset != event["origin_timestamp"]:
            raise ValueError(f"Invalid timestamp/offset mapping: {name}")
        info = inspect_image(folder / name, record["sha256"])
        if any(info[key] != record[key] for key in ["bytes", "width", "height"]):
            raise ValueError(f"Image metadata changed: {name}")
        sizes.add((info["width"], info["height"]))
        total_bytes += info["bytes"]
    if len(sizes) != 1:
        raise ValueError(f"Camera dimensions changed within event: {folder}")
    if sha256(folder / "source_index.html") != manifest["source_index_sha256"]:
        raise ValueError(f"Source index snapshot changed: {folder}")
    parse_index((folder / "source_index.html").read_bytes(), event)
    width, height = sizes.pop()
    return {"event": event["key"], "event_id": event["event_id"], "verified_images": 81,
            "total_bytes": total_bytes, "width": width, "height": height,
            "offset_seconds": [-2400, 2400], "interval_seconds": 60,
            "pre_appearance_images": 40, "appearance_images": 1, "post_appearance_images": 40}


def download_one(folder, event, name):
    target = folder / name
    if target.exists():
        return name, inspect_image(target)
    for attempt in range(3):
        temporary = folder / f".{name}.{uuid.uuid4().hex}.part"
        if not temporary.resolve().is_relative_to((ROOT / "data").resolve()):
            raise ValueError("Temporary file escaped the project data directory.")
        try:
            data = fetch_bytes(CDN + event["event_id"] + "/" + quote(name, safe=""))
            with temporary.open("xb") as stream:
                stream.write(data)
            info = inspect_image(temporary)
            temporary.rename(target)
            return name, info
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
        finally:
            temporary.unlink(missing_ok=True)  # Only this request's temporary D: file.


def download_event(config, event, workers):
    folder = event_directory(config, event)
    if (folder / "manifest.json").exists():
        result = verify_event(config, event)
        print(f"[{event['key']}] existing complete sequence verified", flush=True)
        return result
    index_url = CDN + event["event_id"] + "/index.html"
    index_data = fetch_bytes(index_url)
    names = parse_index(index_data, event)
    folder.mkdir(parents=True, exist_ok=True)
    snapshot = folder / "source_index.html"
    if not snapshot.exists():
        snapshot.write_bytes(index_data)
    parse_index(snapshot.read_bytes(), event)
    information = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(download_one, folder, event, name) for name in names]
        for count, job in enumerate(as_completed(jobs), 1):
            name, info = job.result()
            information[name] = info
            if count % 10 == 0 or count == 81:
                print(f"[{event['key']}] {count}/81 JPEGs downloaded or verified", flush=True)
    if sorted(path.name for path in folder.glob("*.jpg")) != names:
        raise ValueError(f"Unexpected existing images: {folder}")
    if len({(value['width'], value['height']) for value in information.values()}) != 1:
        raise ValueError(f"Inconsistent dimensions in {folder}")
    captured = datetime.now(timezone.utc).isoformat()
    records = []
    for name in names:
        timestamp, offset = map(int, NAME.fullmatch(name).groups())
        records.append({"filename": name, "timestamp": timestamp, "offset_seconds": offset,
                        "url": CDN + event["event_id"] + "/" + quote(name, safe=""), **information[name]})
    manifest = {"event": event["event_id"], "index_url": index_url,
                "downloaded_at_utc": captured, "source_credit": "HPWREN / University of California San Diego",
                "count": 81, "interval_seconds": 60, "source_index_sha256": sha256(snapshot), "files": records}
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = verify_event(config, event)
    verification = {"verified_at_utc": captured, "decoded_all_images": True, **result}
    (folder / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (folder / "SOURCE.md").write_text(
        f"# FIgLib {event['name']} 完整事件\n\n"
        f"- 来源：HPWREN / University of California San Diego，FIgLib。\n"
        f"- 官方目录：{index_url}\n"
        f"- 事件：{event['event_id']}；日期：{event['date']}；机位：{event['camera']}。\n"
        f"- 下载时间（UTC）：{captured}。\n"
        f"- 81张原始彩色JPEG，{result['width']}×{result['height']}，每60秒一张。\n"
        f"- 时间范围−2400至+2400秒；0为官方标记的首次可见烟羽，不是点火时刻。\n"
        f"- 出现前40张、出现时刻1张、出现后40张，未裁剪、增强或改名。\n"
        f"- 每张已完整解码；采集时SHA256和尺寸见manifest.json，不是官方校验和。\n"
        f"- source_index.html为官方目录快照；verification.json为本次离线核验记录。\n"
        f"- 选取发生在模型推理之前；与Whittier及另外两起事件的名称、日期和机位不同。\n"
        f"- 与预训练数据是否重叠尚未核实；未提供人工目标框，需核查其他烟雾。\n"
        f"- 衍生成果须注明HPWREN：https://www.hpwren.ucsd.edu/ 。\n",
        encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", choices=["all", "alison", "west", "cranston"], default="all", type=str.lower)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    config = load_protocol()
    selected = select_events(config, args.event)
    results = [verify_event(config, event) if args.verify_only else download_event(config, event, args.workers) for event in selected]
    print(json.dumps({"events": results, "verified_images": sum(row["verified_images"] for row in results),
                      "total_bytes": sum(row["total_bytes"] for row in results)}, indent=2))


if __name__ == "__main__":
    main()
