"""Download and verify the 81 original JPEGs of the FIgLib Whittier event."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
from urllib.parse import quote, unquote, urljoin, urlsplit
from urllib.request import Request, urlopen
import uuid


ROOT = Path(__file__).resolve().parents[3]
EVENT = "20170708_Whittier_syp-n-mobo-c"
BASE_URL = f"https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/{EVENT}/"
INDEX_URL = BASE_URL + "index.html"
NAME = re.compile(r"^(\d{10})_([+-]\d{5})\.jpg$")
ORIGIN = 1499546263
EXPECTED_NAMES = [f"{ORIGIN + offset}_{offset:+06d}.jpg" for offset in range(-2400, 2401, 60)]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path):
    from PIL import Image

    with Image.open(path) as image:
        if image.format != "JPEG" or image.size != (3072, 2048):
            raise ValueError(f"Unexpected image format/dimensions: {path}")
        image.verify()
    with Image.open(path) as image:
        image.load()  # Decode the complete image; truncated JPEGs must fail.
    return {"bytes": Path(path).stat().st_size, "sha256": sha256(path),
            "width": 3072, "height": 2048}


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.names = set()

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href", "")
        target = urlsplit(urljoin(INDEX_URL, href))
        name = unquote(target.path.rsplit("/", 1)[-1])
        if (target.scheme == "https" and target.netloc == "cdn.hpwren.ucsd.edu"
                and target.path.startswith(urlsplit(BASE_URL).path) and NAME.fullmatch(name)):
            self.names.add(name)


def get_bytes(url):
    request = Request(url, headers={"User-Agent": "AIC-seek/1.0 (FIgLib research download)"})
    with urlopen(request, timeout=60) as response:
        final = urlsplit(response.url)
        if final.scheme != "https" or final.netloc != "cdn.hpwren.ucsd.edu":
            raise ValueError(f"Unexpected redirect: {response.url}")
        return response.read()


def download_one(name, output):
    target = output / name
    if target.exists():
        return name, "reused", inspect_image(target)
    last_error = None
    for attempt in range(3):
        temporary = output / f".{name}.{uuid.uuid4().hex}.part"
        try:
            # The official index encodes '+' in image URLs as %2B.
            data = get_bytes(BASE_URL + quote(name, safe=""))
            with temporary.open("xb") as stream:
                stream.write(data)
            info = inspect_image(temporary)
            if target.exists():
                raise FileExistsError(f"Destination appeared during download: {target}")
            temporary.rename(target)
            return name, "downloaded", info
        except Exception as error:
            last_error = error
            if attempt < 2:
                time.sleep(attempt + 1)
        finally:
            # Only this task's temporary file, beside the D: destination.
            temporary.unlink(missing_ok=True)
    raise RuntimeError(f"Failed to download {name}: {last_error}") from last_error


def verify(output, original_ten=None):
    images = sorted(p.name for p in output.glob("*.jpg"))
    if images != EXPECTED_NAMES:
        missing = sorted(set(EXPECTED_NAMES) - set(images))
        extra = sorted(set(images) - set(EXPECTED_NAMES))
        raise ValueError(f"Image set mismatch: missing={missing}, extra={extra}")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    expected_hashes = {row["filename"]: row["sha256"] for row in manifest["files"]} if manifest else {}
    rows = []
    comparisons = []
    for name in images:
        info = inspect_image(output / name)
        if expected_hashes and expected_hashes.get(name) != info["sha256"]:
            raise ValueError(f"Manifest hash mismatch: {name}")
        match = NAME.fullmatch(name)
        offset = int(match[2])
        if int(match[1]) - offset != ORIGIN:
            raise ValueError(f"Timestamp/offset mismatch: {name}")
        rows.append({"filename": name, "url": BASE_URL + quote(name, safe=""),
                     "timestamp": int(match[1]), "offset_seconds": offset, **info})
        if original_ten is not None and (original_ten / name).is_file():
            same = sha256(original_ten / name) == info["sha256"]
            comparisons.append({"filename": name, "identical_to_original_ten": same})
            if not same:
                raise ValueError(f"Downloaded image differs from the original sample: {name}")
    if original_ten is not None and len(comparisons) != 10:
        raise ValueError(f"Expected comparisons with all 10 old images, got {len(comparisons)}")
    return rows, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/figlib_whittier_full_81")
    parser.add_argument("--compare-ten", type=Path, default=ROOT / "data/figlib_whittier_smoke_10")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--verify-only", action="store_true", help="Read and verify local files without network or writes.")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    old = args.compare_ten.resolve()
    if args.verify_only:
        if not (output / "manifest.json").is_file():
            raise SystemExit("manifest.json is required for offline hash verification.")
        rows, comparisons = verify(output, old)
        print(json.dumps({"verified_images": len(rows), "total_bytes": sum(r["bytes"] for r in rows),
                          "offset_seconds": [-2400, 2400], "old_images_identical": len(comparisons)}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    index_bytes = get_bytes(INDEX_URL)
    links = Links()
    links.feed(index_bytes.decode("utf-8"))
    if sorted(links.names) != EXPECTED_NAMES:
        raise SystemExit("Official index does not match the expected 81-frame event; no images downloaded.")
    (output / "source_index.html").write_bytes(index_bytes)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(download_one, name, output) for name in EXPECTED_NAMES]
        for count, future in enumerate(as_completed(pending), 1):
            name, action, info = future.result()
            print(f"[{count:02d}/81] {action}: {name} ({info['bytes']} bytes)", flush=True)
    rows, comparisons = verify(output, old)
    captured = datetime.now(timezone.utc).isoformat()
    manifest = {"event": EVENT, "index_url": INDEX_URL, "downloaded_at_utc": captured,
                "source_credit": "HPWREN / University of California San Diego",
                "count": len(rows), "interval_seconds": 60, "files": rows}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    verification = {"verified_at_utc": captured, "verified_images": len(rows),
                    "total_bytes": sum(row["bytes"] for row in rows),
                    "pre_appearance_images": 40, "appearance_images": 1,
                    "post_appearance_images": 40, "interval_seconds": 60,
                    "decoded_all_images": True, "original_ten_comparisons": comparisons}
    (output / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "SOURCE.md").write_text(
        f"# FIgLib Whittier完整81帧\n\n"
        f"- 来源：HPWREN / University of California San Diego，FIgLib。\n"
        f"- 数据说明：https://www.hpwren.ucsd.edu/FIgLib/\n"
        f"- 官方事件目录：{INDEX_URL}\n"
        f"- 事件：{EVENT}；同一固定相机。\n"
        f"- 下载时间（UTC）：{captured}。\n"
        f"- 81张原始JPEG，3072×2048，每60秒一张，偏移−2400至+2400秒。\n"
        f"- 偏移0表示首次可见烟羽，不是真实点火时刻。\n"
        f"- 40张位于该烟羽出现前，1张为出现时刻，40张位于出现后。\n"
        f"- 未裁剪、增强或改名；逐图完整解码，SHA256见manifest.json。\n"
        f"- 旧10帧与新下载对应文件逐字节哈希一致，详见verification.json。\n"
        f"- source_index.html是下载时保存的官方目录快照。\n"
        f"- 本目录没有人工框或掩码标注；出现前片段须人工核查其他烟雾。\n"
        f"- 与预训练数据是否重叠尚未核实；单事件不能证明泛化或长期误报率。\n"
        f"- 衍生成果须注明HPWREN来源：https://www.hpwren.ucsd.edu/ 。其他条件以官方为准。\n",
        encoding="utf-8")
    print(json.dumps({"saved_to": str(output), "images": len(rows),
                      "total_bytes": verification["total_bytes"], "old_images_identical": len(comparisons)}, indent=2))


if __name__ == "__main__":
    main()
