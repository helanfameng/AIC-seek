"""烟雾瞭望：给别人看固定相机照片用的本地页面。"""

import base64
import csv
import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
DATA = ROOT / "data" / "figlib_whittier_full_81"
BATCH = ROOT / "experiments" / "exp003_whittier_full_replay" / "outputs" / "exp003_20261019_224724"
MODEL = ROOT / "models" / "v0.6.0" / "model.zip"
RUNNER = ROOT / "experiments" / "exp001_pyronear_baseline" / "scripts" / "run_sequence.py"
CACHE = ROOT / ".cache" / "web_preview"
SESSIONS = ROOT / ".cache" / "web_sessions"
BOARD = ROOT / ".cache" / "web_board.json"
CHAT = ROOT / ".cache" / "chat_photos"
HOST = "127.0.0.1"
PORT = 8765
CONTEXT = 1.5
MODES = {"standard": "full_baseline", "sensitive": "low_threshold"}
MODE_LABEL = {"standard": "标准", "sensitive": "更敏感"}
IMAGE_EXT = {".jpg", ".jpeg", ".png"}
NAME_OK = re.compile(r"^[^<>:\"/\\|?*\x00-\x1f]+$")
DEMO_NAME = re.compile(r"^\d{10}_[+-]\d{5}\.jpg$", re.IGNORECASE)
SIZES = {160, 1440}
MAX_UPLOAD = 20 * 1024 * 1024
MAX_FILES = 40

demo_lock = threading.Lock()
demo_cache = None
sessions_lock = threading.Lock()
sessions = {}
board_lock = threading.Lock()
live_lock = threading.Lock()
live_cache = {}


def minute(offset):
    return int(round(offset / 60))


def clock_phrase(offset):
    value = minute(offset)
    if value == 0:
        return "烟刚可见的这一分钟"
    if value > 0:
        return f"烟可见后 {value} 分钟"
    return f"烟可见前 {-value} 分钟"


def read_csv(group):
    path = BATCH / group / "summary.csv"
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            score = raw.get("max_probability") or ""
            age = raw.get("winner_observation_age_seconds") or ""
            rows.append({
                "filename": raw["filename"],
                "offset": int(float(raw["offset_seconds"])),
                "candidates": int(float(raw["current_candidate_count"] or 0)),
                "kept": int(float(raw["kept_tube_count"] or 0)),
                "alarm": raw["is_positive"] == "True",
                "score": float(score) if score else None,
                "age": float(age) if age else None,
            })
    return rows


def load_steps(group):
    found = {}
    path = BATCH / group / "steps.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            found[row["summary"]["filename"]] = row
    return found


def pack(rect, kind):
    x, y, width, height = rect
    return {"kind": kind, "x": x, "y": y, "w": width, "h": height}


def norm_rect(box, width, height):
    cx, cy, bw, bh = box
    return [(cx - bw / 2) * width, (cy - bh / 2) * height, bw * width, bh * height]


def crop_rect(box, width, height):
    cx, cy, bw, bh = box
    bw *= CONTEXT
    bh *= CONTEXT
    side = max(bw * width, bh * height)
    half = side / 2
    cx_px = cx * width
    cy_px = cy * height
    x0 = max(0, int(round(cx_px - half)))
    y0 = max(0, int(round(cy_px - half)))
    x1 = min(width, int(round(cx_px + half)))
    y1 = min(height, int(round(cy_px + half)))
    return [x0, y0, x1 - x0, y1 - y0]


def shapes_for_step(step, width, height):
    stem = Path(step["summary"]["filename"]).stem
    detections = []
    for item in step.get("raw_detections") or []:
        if item.get("frame_id") == stem:
            detections = item.get("detections") or []
    shapes = [pack(norm_rect([item["cx"], item["cy"], item["w"], item["h"]], width, height), "suspect")
              for item in detections]
    kept = (((step.get("prediction") or {}).get("details") or {}).get("tubes") or {}).get("kept") or []
    ids = step.get("effective_frame_ids") or []
    for tube in kept:
        entries = []
        for entry in tube.get("entries") or []:
            index = entry.get("frame_idx")
            if isinstance(index, int) and 0 <= index < len(ids) and ids[index] == stem and entry.get("bbox"):
                entries.append(entry)
        if not entries:
            continue
        for entry in entries:
            shapes.append(pack(norm_rect(entry["bbox"], width, height), "gap" if entry.get("is_gap") else "track"))
        window = tube.get("stabilized_window") or entries[0]["bbox"]
        shapes.append(pack(crop_rect(window, width, height), "crop"))
    return shapes


def shapes_from_prediction(prediction, filename, width, height):
    names = prediction.get("frame_names") or []
    wanted = Path(filename).name
    try:
        index = names.index(wanted)
    except ValueError:
        return []
    shapes = []
    kept = (((prediction.get("details") or {}).get("tubes") or {}).get("kept") or [])
    for tube in kept:
        entries = [entry for entry in tube.get("entries") or []
                   if entry.get("frame_idx") == index and entry.get("bbox")]
        if not entries:
            continue
        for entry in entries:
            shapes.append(pack(norm_rect(entry["bbox"], width, height), "gap" if entry.get("is_gap") else "track"))
        window = tube.get("stabilized_window") or entries[0]["bbox"]
        shapes.append(pack(crop_rect(window, width, height), "crop"))
    return shapes


def first_where(rows, pred):
    for row in rows:
        if pred(row):
            return row["offset"]
    return None


def sequence_rows():
    rows = []
    for path in DATA.glob("*.jpg"):
        match = re.fullmatch(r"(\d{10})_([+-]\d{5})\.jpg", path.name, re.IGNORECASE)
        if match is None:
            continue
        rows.append((int(match.group(2)), path.name))
    rows.sort()
    return rows


def blank_stats():
    return {"candidates": 0, "kept": 0, "alarm": False, "score": None, "age": None, "shapes": [], "pending": True}


def build_demo():
    rows = sequence_rows()
    if not rows:
        raise RuntimeError("没有原图。")
    with Image.open(DATA / rows[0][1]) as image:
        width, height = image.size
    frames = [{
        "filename": name,
        "offset": offset,
        "standard": blank_stats(),
        "sensitive": blank_stats(),
    } for offset, name in rows]
    modes = {
        mode: {"label": MODE_LABEL[mode], "alarm_offset": None, "suspect_offset": None}
        for mode in MODES
    }
    return {
        "place": "圣伊内斯峰北",
        "when": "",
        "source": "固定相机原图",
        "width": width,
        "height": height,
        "intro": "正在对原图做自动检查。",
        "compare": "",
        "modes": modes,
        "frames": frames,
    }


def get_demo():
    global demo_cache
    with demo_lock:
        if demo_cache is None:
            demo_cache = build_demo()
        return demo_cache


WATCH = ROOT / ".cache" / "web_watch.json"
watch_lock = threading.Lock()
watch_state = {"status": "idle", "done": 0, "total": 0, "message": "", "frames": {}}


def watch_modes(order):
    checked = [item for item in order if item["filename"] in watch_state["frames"]]
    def first(predicate):
        found = next((item["offset"] for item in checked if predicate(item)), None)
        return found
    stats = lambda item: watch_state["frames"][item["filename"]]
    return {
        mode: {
            "label": MODE_LABEL[mode],
            "alarm_offset": first(lambda item: item["offset"] >= 0 and stats(item)["alarm"]),
            "suspect_offset": first(lambda item: item["offset"] >= 0 and stats(item)["candidates"] > 0),
        }
        for mode in MODES
    }


def watch_snapshot():
    with watch_lock:
        order = [{"offset": offset, "filename": name} for offset, name in sequence_rows()]
        frames = []
        for item in order:
            stats = watch_state["frames"].get(item["filename"])
            if stats is None:
                continue
            frames.append({"filename": item["filename"], "standard": stats, "sensitive": stats})
        return {
            "status": watch_state["status"],
            "done": watch_state["done"],
            "total": watch_state["total"],
            "message": watch_state["message"],
            "modes": watch_modes(order),
            "frames": frames,
        }


def stats_for_window(result, detections, frame_index, width, height):
    details = result.details
    if hasattr(details, "model_dump"):
        details = details.model_dump()
    kept = (((details or {}).get("tubes") or {}).get("kept") or [])
    shapes = [pack(norm_rect([item.cx, item.cy, item.w, item.h], width, height), "suspect") for item in detections.detections]
    touching = []
    saw_real = False
    for tube in kept:
        entries = [entry for entry in tube.get("entries") or [] if entry.get("frame_idx") == frame_index and entry.get("bbox")]
        if not entries:
            continue
        touching.append(tube)
        for entry in entries:
            shapes.append(pack(norm_rect(entry["bbox"], width, height), "gap" if entry.get("is_gap") else "track"))
            saw_real = saw_real or not entry.get("is_gap")
        window = tube.get("stabilized_window") or entries[0]["bbox"]
        shapes.append(pack(crop_rect(window, width, height), "crop"))
    pool = touching or kept
    probabilities = [float(tube["probability"]) for tube in pool if isinstance(tube.get("probability"), (int, float))]
    alarm = bool(result.is_positive)
    return {
        "candidates": len(detections.detections),
        "kept": len(touching),
        "alarm": alarm,
        "score": max(probabilities) if probabilities else None,
        "age": 0 if saw_real or not alarm else 60,
        "shapes": shapes,
        "pending": False,
    }


def remember_watch(filename, stats, done, total):
    with watch_lock:
        watch_state["frames"][filename] = stats
        watch_state["done"] = done
        watch_state["total"] = total
        watch_state["status"] = "done" if done >= total else "running"
        watch_state["message"] = "原图已检查完" if done >= total else f"正在检查原图 {done} / {total}"


def load_watch_cache(names):
    if not WATCH.is_file() or not MODEL.is_file():
        return False
    try:
        saved = json.loads(WATCH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    cached = saved.get("frames") or {}
    if saved.get("model_size") != MODEL.stat().st_size or saved.get("files") != names or len(cached) != len(names):
        return False
    with watch_lock:
        watch_state["frames"] = cached
        watch_state["done"] = len(names)
        watch_state["total"] = len(names)
        watch_state["status"] = "done"
        watch_state["message"] = "原图已检查完"
    return True


def save_watch_cache(names):
    with watch_lock:
        payload = {
            "model_size": MODEL.stat().st_size,
            "files": names,
            "frames": watch_state["frames"],
        }
    WATCH.parent.mkdir(parents=True, exist_ok=True)
    temporary = WATCH.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(WATCH)


def run_watch():
    rows = sequence_rows()
    names = [name for _, name in rows]
    with watch_lock:
        watch_state["total"] = len(names)
        watch_state["status"] = "running"
        watch_state["message"] = "正在加载模型"
    if load_watch_cache(names):
        print("原图检查已有结果", len(names))
        return
    if not MODEL.is_file():
        with watch_lock:
            watch_state["status"] = "error"
            watch_state["message"] = "还不能自动检查。请把模型放到 models/v0.6.0/model.zip，不用解压。"
        return
    try:
        import torch
        from temporal_model.core.model import BboxTubeTemporalModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = BboxTubeTemporalModel.from_package(MODEL, device=device)
        window = int(model._cfg["classifier"].get("max_frames") or 20)
        loaded = model.load_sequence([DATA / name for name in names])
        with Image.open(DATA / names[0]) as image:
            width, height = image.size
        detected = {}
        print("开始检查原图", len(loaded), device)
        for end, frame in enumerate(loaded):
            found = model.detect([frame])[0]
            detected[frame.frame_id] = found
            start = max(0, end - window + 1)
            chunk = loaded[start:end + 1]
            result = model.predict(chunk, frame_detections=detected)
            stats = stats_for_window(result, found, len(chunk) - 1, width, height)
            remember_watch(names[end], stats, end + 1, len(names))
            if stats["alarm"] or stats["candidates"]:
                print("原图", names[end], "可疑", stats["candidates"], "留下", stats["kept"], "建议查看" if stats["alarm"] else "继续看")
        save_watch_cache(names)
        print("原图检查完成", len(names), flush=True)
    except Exception as exc:
        message = "自动检查没有完成。"
        detail = str(exc)
        if getattr(exc, "winerror", None) == 4551 or "4551" in detail or "c10.dll" in detail:
            message = "本机拦住了模型运行库，自动检查没有开始。"
        with watch_lock:
            watch_state["status"] = "error"
            watch_state["message"] = message
        print("原图检查没有完成", type(exc).__name__, exc, flush=True)


def cleanup_sessions():
    if not SESSIONS.exists():
        return
    cutoff = time.time() - 2 * 24 * 3600
    for child in SESSIONS.iterdir():
        try:
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            continue


def new_session():
    identifier = uuid.uuid4().hex
    folder = SESSIONS / identifier / "frames"
    folder.mkdir(parents=True, exist_ok=True)
    session = {
        "id": identifier,
        "dir": SESSIONS / identifier,
        "files": [],
        "status": "collecting",
        "error": "",
        "detail": "",
        "retry": None,
        "started": None,
        "result": None,
        "proc": None,
    }
    with sessions_lock:
        sessions[identifier] = session
    return session


def session_payload(session):
    frames = []
    for item in sorted(session["files"], key=lambda row: row["name"].lower()):
        shapes = []
        if session.get("result"):
            shapes = shapes_from_prediction(session["result"], item["name"], item["width"], item["height"])
        frames.append({
            "filename": item["name"],
            "width": item["width"],
            "height": item["height"],
            "shapes": shapes,
        })
    verdict = None
    if session.get("result"):
        verdict = {
            "alarm": bool(session["result"].get("is_smoke")),
            "score": session["result"].get("max_kept_probability"),
            "kept": session["result"].get("kept_tube_count") or 0,
        }
    return {
        "id": session["id"],
        "status": session["status"],
        "error": session["error"],
        "detail": session["detail"],
        "retry": session["retry"],
        "started": session["started"],
        "file_count": len(frames),
        "verdict": verdict,
        "frames": frames,
    }


def save_chat_photo(name, data):
    filename = Path(str(name or "")).name
    suffix = Path(filename).suffix.lower()
    if not filename or suffix not in IMAGE_EXT or not NAME_OK.match(filename):
        raise ValueError("请放入 JPG 或 PNG。")
    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("请放入 JPG 或 PNG。")
            width, height = image.size
    except (UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError("这张照片打不开，或是尺寸太大。") from exc
    if width < 32 or height < 32:
        raise ValueError("这张照片太小了。")
    CHAT.mkdir(parents=True, exist_ok=True)
    stored = f"{uuid.uuid4().hex[:8]}_{Path(filename).stem[:40]}{suffix}"
    (CHAT / stored).write_bytes(data)
    return stored


def chat_path(name):
    filename = Path(str(name or "")).name
    if filename != str(name or "") or not NAME_OK.match(filename) or Path(filename).suffix.lower() not in IMAGE_EXT:
        return None
    root = CHAT.resolve()
    path = (root / filename).resolve()
    if path.parent != root:
        return None
    return path


def save_upload(session, name, data):
    if session["status"] == "running":
        raise ValueError("正在检查，先等这一组结束。")
    if len(session["files"]) >= MAX_FILES:
        raise ValueError(f"一次最多 {MAX_FILES} 张。")
    if not name or not NAME_OK.match(name) or Path(name).suffix.lower() not in IMAGE_EXT:
        raise ValueError("请放入 JPG 或 PNG，文件名里不要带斜杠。")
    if any(item["name"].lower() == name.lower() for item in session["files"]):
        raise ValueError("已经有一张同名照片了。")
    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            width, height = image.size
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("请放入 JPG 或 PNG。")
    except (UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError("这张照片打不开，或是尺寸太大。") from exc
    if width < 32 or height < 32:
        raise ValueError("这张照片太小了。")
    (session["dir"] / "frames" / name).write_bytes(data)
    session["files"].append({"name": name, "width": width, "height": height})
    session["result"] = None
    session["status"] = "collecting"
    session["error"] = ""
    session["detail"] = ""
    session["retry"] = None


def delete_upload(session, name):
    if session["status"] == "running":
        raise ValueError("正在检查，先等这一组结束。")
    path = session["dir"] / "frames" / Path(name).name
    if path.exists():
        path.unlink()
    session["files"] = [item for item in session["files"] if item["name"] != Path(name).name]
    session["result"] = None
    session["status"] = "collecting"
    session["error"] = ""


def run_check(session, device):
    frames = session["dir"] / "frames"
    output = session["dir"] / "predictions.json"
    if output.exists():
        output.unlink()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    command = [sys.executable, str(RUNNER), "--sequence", str(frames), "--model", str(MODEL),
               "--output", str(output), "--device", device]
    session["log"] = []
    try:
        process = subprocess.Popen(
            command, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=env,
        )
    except OSError as exc:
        session["status"] = "error"
        session["error"] = "没有能启动检查。"
        session["detail"] = str(exc)
        return
    session["proc"] = process
    for line in process.stdout:
        session["log"].append(line.rstrip())
        if len(session["log"]) > 80:
            del session["log"][:-80]
    code = process.wait()
    session["proc"] = None
    tail = "\n".join(session["log"][-12:])
    if code == 0 and output.exists():
        session["result"] = json.loads(output.read_text(encoding="utf-8"))
        session["status"] = "done"
        session["error"] = ""
        session["detail"] = ""
        session["retry"] = None
        return
    session["status"] = "error"
    session["detail"] = tail
    if "CUDA is unavailable" in tail:
        session["error"] = "没有用上显卡。可以改用本机计算，会慢一些。"
        session["retry"] = "cpu"
    elif "Model package not found" in tail:
        session["error"] = "还没有模型文件，所以不能自动检查。"
        session["retry"] = None
    else:
        session["error"] = "检查没有完成。"
        session["retry"] = None


def start_check(session, device):
    if device not in {"cuda", "cpu"}:
        device = "cuda"
    if not MODEL.is_file():
        raise ValueError("还不能自动检查。请把模型放到 models/v0.6.0/model.zip，不用解压。")
    if not session["files"]:
        raise ValueError("先放进照片。")
    if session["status"] == "running":
        raise ValueError("这一组已经在检查了。")
    session["status"] = "running"
    session["error"] = ""
    session["detail"] = ""
    session["retry"] = None
    session["result"] = None
    session["started"] = time.time()
    threading.Thread(target=run_check, args=(session, device), daemon=True).start()


def cached_jpeg(source, key, render):
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / f"{key}.jpg"
    if not dest.exists() or dest.stat().st_mtime < source.stat().st_mtime:
        render(source, dest)
    return dest.read_bytes()


def render_size(source, dest, size):
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((size, size * 2), Image.Resampling.LANCZOS)
        image.save(dest, "JPEG", quality=78 if size <= 160 else 84)


def render_crop(source, dest, box):
    with Image.open(source) as image:
        image = image.convert("RGB")
        width, height = image.size
        x, y, bw, bh = box
        side_limit = max(bw, bh, 220)
        if bw >= 220 and bh >= 220:
            cx, cy = x + bw / 2, y + bh / 2
            bw, bh = bw * 1.08, bh * 1.08
            x, y = cx - bw / 2, cy - bh / 2
        else:
            cx, cy = x + bw / 2, y + bh / 2
            x, y = cx - side_limit / 2, cy - side_limit / 2
            bw = bh = side_limit
        x0 = max(0, int(round(x)))
        y0 = max(0, int(round(y)))
        x1 = min(width, int(round(x + bw)))
        y1 = min(height, int(round(y + bh)))
        if x1 - x0 < 2 or y1 - y0 < 2:
            raise ValueError("圈出的范围无效。")
        crop = image.crop((x0, y0, x1, y1))
        crop.thumbnail((520, 520), Image.Resampling.LANCZOS)
        crop.save(dest, "JPEG", quality=86)


def demo_file(name):
    if not DEMO_NAME.match(name):
        return None
    path = DATA / name
    return path if path.is_file() else None


def session_file(identifier, name):
    if not re.fullmatch(r"[a-f0-9]{32}", identifier):
        return None
    with sessions_lock:
        session = sessions.get(identifier)
    folder = SESSIONS / identifier / "frames"
    path = folder / Path(name).name
    if session is None:
        known = path.is_file() and path.parent == folder
    else:
        known = any(item["name"] == path.name for item in session["files"])
    if known and path.is_file():
        return path
    return None


def wind_name(degrees):
    names = ["北风", "东北风", "东风", "东南风", "南风", "西南风", "西风", "西北风"]
    try:
        value = float(degrees)
    except (TypeError, ValueError):
        return "风向不明"
    return names[int((value + 22.5) // 45) % 8]


def fresh_board():
    demo = get_demo()
    frames = demo["frames"]
    appear = next((item for item in frames if item["offset"] == 0), frames[0])
    alarm = next((item for item in frames if item["offset"] >= 0 and item["standard"]["alarm"]), appear)
    first = frames[0]
    now = int(time.time() * 1000)
    score = alarm["standard"].get("score")
    pct = round(score * 100) if isinstance(score, (int, float)) else 0
    people = []

    def order(frame, index, **extra):
        status = extra.get("status") or "doing"
        if status == "pending":
            status = "doing"
        started = extra.get("startedAt")
        if started is None and "startedAt" not in extra:
            started = now
        base = {
            "id": uuid.uuid4().hex[:8],
            "siteId": "syp-n",
            "title": extra.get("title") or "核对圣伊内斯峰北烟雾",
            "level": extra.get("level") or "alarm",
            "summary": extra.get("summary") or "这一张比较像烟雾，建议有人看一眼。",
            "score": extra.get("score") or "",
            "advice": extra.get("advice") or "智能助手：圣伊内斯峰北已到建议查看。请核对圈出的位置，确认是烟、云，还是尘和光斑，再回传。",
            "status": status,
            "assigneeId": extra.get("assigneeId") or "",
            "action": "",
            "note": extra.get("note") or "",
            "support": False,
            "createdAt": extra.get("createdAt") or now,
            "startedAt": started,
            "finishedAt": extra.get("finishedAt"),
            "urgedAt": 0,
            "afterShot": False,
            "media": {"kind": "demo", "filename": frame["filename"], "index": index},
            "events": [{"at": extra.get("createdAt") or now, "who": "中控", "text": "生成工单"}],
            "messages": [],
        }
        return base

    alarm_index = frames.index(alarm)
    appear_index = frames.index(appear)
    return {
        "rev": 1,
        "people": people,
        "sites": [],
        "cameras": [],
        "orders": [
            order(alarm, alarm_index, score=f"把握大约 {pct}%"),
            order(
                appear, appear_index, title="烟刚可见，持续观察", level="watch", status="doing",
                summary="烟刚刚能看见，先连续看，还没到建议出动。",
                advice="智能助手：圣伊内斯峰北发现可疑。同一处连续出现，继续看下一分钟是否留在原处。",
                createdAt=now - 8 * 60000, startedAt=now - 8 * 60000,
            ),
            order(
                first, 0, title="烟出现前复核", level="ok", status="done",
                summary="可见之前没有烟。", note="可见之前没有烟。",
                advice="智能助手：这一张没有需要出动的标记。",
                createdAt=now - 40 * 60000, startedAt=now - 40 * 60000, finishedAt=now - 39 * 60000,
            ),
        ],
    }


def write_board(board):
    BOARD.parent.mkdir(parents=True, exist_ok=True)
    temporary = BOARD.with_suffix(".tmp")
    temporary.write_text(json.dumps(board, ensure_ascii=False), encoding="utf-8")
    temporary.replace(BOARD)


def read_board():
    if not BOARD.is_file():
        board = fresh_board()
        write_board(board)
        return board
    return json.loads(BOARD.read_text(encoding="utf-8"))


def load_board():
    with board_lock:
        return read_board()


def clean_people(people):
    cleaned = []
    seen = set()
    for item in people:
        if not isinstance(item, dict):
            continue
        phone = "".join(ch for ch in str(item.get("phone") or "") if ch.isdigit())
        name = str(item.get("name") or "").strip()[:20]
        if not name or not phone.startswith("1") or len(phone) != 11 or phone in seen:
            continue
        seen.add(phone)
        try:
            history = int(item.get("historyDone") or 0)
        except (TypeError, ValueError):
            history = 0
        skill = str(item.get("skill") or "现场处置").strip()[:12] or "现场处置"
        cleaned.append({
            "id": str(item.get("id") or phone)[:16],
            "name": name,
            "phone": phone,
            "skill": skill,
            "region": str(item.get("region") or "").strip()[:20],
            "historyDone": max(0, history),
            "added": bool(item.get("added")),
            "password": str(item.get("password") or "")[:32],
        })
    return cleaned


BUILTIN_SITE_IDS = {"syp-n", "syp-s", "cach", "fig", "lac", "gib", "ref", "cam"}


def bounded_number(value, low, high):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not (low <= number <= high):
        return None
    return number


def clean_sites(sites):
    cleaned = []
    seen = set()
    for item in sites:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()[:20]
        code = str(item.get("code") or "").strip()[:12]
        if not name or not code or code in seen:
            continue
        lat = bounded_number(item.get("lat"), -90, 90)
        lng = bounded_number(item.get("lng"), -180, 180)
        elev = bounded_number(item.get("elev"), -500, 9000)
        if lat is None or lng is None or elev is None:
            continue
        seen.add(code)
        cleaned.append({
            "id": str(item.get("id") or code)[:16],
            "code": code,
            "name": name,
            "line": str(item.get("line") or "自加点位").strip()[:20] or "自加点位",
            "region": str(item.get("region") or "").strip()[:20],
            "lat": lat,
            "lng": lng,
            "elev": int(round(elev)),
            "live": False,
            "kind": "瞭望点",
            "tone": "ok",
            "status": "正常",
            "detail": "手动添加的点位，没有连续画面。",
        })
    return cleaned


def clean_ip(value):
    parts = str(value or "").strip().split(".")
    if len(parts) != 4:
        return ""
    numbers = []
    for part in parts:
        if not part.isdigit() or len(part) > 3:
            return ""
        number = int(part)
        if number > 255:
            return ""
        numbers.append(str(number))
    return ".".join(numbers)


def clean_cameras(cameras, site_ids):
    cleaned = []
    seen = set()
    for item in cameras:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()[:20]
        site_id = str(item.get("siteId") or "").strip()[:16]
        if not name or site_id not in site_ids:
            continue
        key = (site_id, name)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({
            "id": str(item.get("id") or site_id)[:16],
            "siteId": site_id,
            "name": name,
            "ip": clean_ip(item.get("ip")),
        })
    return cleaned


def settle_orders(orders):
    changed = False
    now = int(time.time() * 1000)
    for order in orders or []:
        if not isinstance(order, dict) or order.get("status") != "pending":
            continue
        order["status"] = "doing"
        if not order.get("startedAt"):
            order["startedAt"] = order.get("createdAt") or now
        changed = True
    return changed


def migrate_board():
    with board_lock:
        board = read_board()
        if settle_orders(board.get("orders")):
            board["rev"] = int(board.get("rev") or 0) + 1
            write_board(board)


def store_board(payload):
    if not isinstance(payload, dict):
        raise ValueError("工单数据不完整。")
    orders = payload.get("orders")
    people = payload.get("people")
    if not isinstance(orders, list) or not isinstance(people, list):
        raise ValueError("工单数据不完整。")
    with board_lock:
        current = read_board()
        if payload.get("rev") != current.get("rev"):
            return current, False
        current["orders"] = orders
        settle_orders(current["orders"])
        current["people"] = clean_people(people)
        if isinstance(payload.get("sites"), list):
            current["sites"] = clean_sites(payload["sites"])
        site_ids = set(BUILTIN_SITE_IDS) | {item["id"] for item in current.get("sites") or []}
        if isinstance(payload.get("cameras"), list):
            current["cameras"] = clean_cameras(payload["cameras"], site_ids)
        current["rev"] = int(current.get("rev") or 0) + 1
        write_board(current)
        return current, True


def reset_board():
    with board_lock:
        board = fresh_board()
        write_board(board)
        return board


def remote_json(url, headers):
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=6) as response:
        return json.loads(response.read().decode("utf-8"))


def live_report(lat, lng):
    key = f"{round(lat, 3)},{round(lng, 3)}"
    now = time.time()
    with live_lock:
        cached = live_cache.get(key)
        if cached and now - cached["at"] < 600:
            return cached["data"]
    weather = None
    place = None
    try:
        forecast = remote_json(
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lng}"
            "&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
            "&wind_speed_unit=ms&timezone=America%2FLos_Angeles",
            {"User-Agent": "AIC-seek/1.0"},
        )
        current = forecast.get("current") or {}
        weather = {
            "temp": f"{round(current['temperature_2m'])}°C",
            "humidity": f"{round(current['relative_humidity_2m'])}%",
            "wind": f"{round(current['wind_speed_10m'])} 米/秒",
            "dir": wind_name(current.get("wind_direction_10m")),
            "source": "Open-Meteo",
        }
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError):
        weather = None
    try:
        found = remote_json(
            "https://nominatim.openstreetmap.org/reverse"
            f"?format=jsonv2&lat={lat}&lon={lng}",
            {"User-Agent": "AIC-seek/1.0 (local smoke watch)"},
        )
        place = {"name": found.get("name") or (found.get("display_name") or "").split(",")[0], "source": "OpenStreetMap"}
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        place = None
    data = {"weather": weather, "place": place}
    with live_lock:
        live_cache[key] = {"at": now, "data": data}
    return data


judge_cache = {}
judge_lock = threading.Lock()


def judge_report(body):
    if not isinstance(body, dict):
        raise ValueError("研判内容不对。")
    status = str(body.get("status") or "")
    if status not in ("发现可疑", "建议查看"):
        raise ValueError("当前没有需要研判的烟雾。")
    site = str(body.get("site") or "瞭望点").strip()[:40] or "瞭望点"
    code = str(body.get("code") or "").strip()[:16]
    try:
        elev = int(body.get("elev") or 0)
        candidates = int(body.get("candidates") or 0)
        kept = int(body.get("kept") or 0)
    except (TypeError, ValueError):
        raise ValueError("研判内容不对。")
    watched = str(body.get("watched") or "").strip()[:40]
    weather = "，".join(
        part for part in (
            str(body.get("temp") or "").strip(),
            f"湿度 {body.get('humidity')}" if body.get("humidity") else "",
            f"{body.get('dir') or ''} {body.get('wind') or ''}".strip(),
        ) if part
    ) or "暂未接通"
    signature = f"{site}|{status}|{weather}|{watched}|noscore"
    refresh = bool(body.get("refresh"))
    with judge_lock:
        cached = None if refresh else judge_cache.get(signature)
    if cached:
        return cached
    key = read_env_value("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("还没有 DeepSeek 密钥。写到 .env 的 DEEPSEEK_API_KEY 后再生成。")
    facts = "\n".join([
        f"点位：{site} {code}".strip(),
        f"海拔：约 {elev} 米",
        f"当前状态：{status}",
        f"这一张标出可疑 {candidates} 处，连续留下来 {kept} 处",
        f"观察进度：{watched or '正在看'}",
        f"现场天气：{weather or '暂未接通'}",
    ])
    payload = json.dumps({
        "model": "deepseek-chat",
        "temperature": 0.4,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是林火瞭望调度中心的值班研判员。固定相机正在实时监测，模型刚给出这一张的检测结果。"
                    "根据给出的事实写专业研判，供值班调度直接执行。"
                    "分析要说明：标记代表什么、连续出现说明什么、风会把烟往哪一带送。"
                    "发现可疑时，方案是继续盯住原处，不要立刻当火场。"
                    "建议查看时，方案要写清派人核对的步骤：看哪一块、核对烟、云、尘或光斑、什么情况升级、什么情况可以继续观察。"
                    "不要写像烟雾的把握、置信度、百分比或分数，这个数字会随画面变动。"
                    "给出的处数、观察进度、气温、湿度、风向风速必须原样使用，不要改成别的数字。"
                    "天气是暂未接通时，不要写气温、湿度和风。"
                    "不要编造明火、过火面积、被困人员，不要提到回放、历史录像或数据集。"
                    '只返回 JSON：{"headline":"不超过20字","analysis":["分析第一段","分析第二段"],"actions":["处置第一步","处置第二步","处置第三步"]}'
                ),
            },
            {"role": "user", "content": facts},
        ],
    }, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = json.loads(response.read().decode("utf-8"))
        content = raw["choices"][0]["message"]["content"]
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            fence = text.rfind("```")
            if fence >= 0:
                text = text[:fence]
        parsed = json.loads(text.strip())
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise ValueError("DeepSeek 密钥没有通过。请核对 .env 里的 DEEPSEEK_API_KEY。") from None
        raise ValueError("研判没有生成，请再试一次。") from None
    except (urllib.error.URLError, TimeoutError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print("研判失败", type(exc).__name__, flush=True)
        raise ValueError("研判没有生成，请再试一次。") from None
    parsed = parsed if isinstance(parsed, dict) else {}
    analysis = parsed.get("analysis")
    if not isinstance(analysis, list):
        analysis = parsed.get("paragraphs")
    if not isinstance(analysis, list):
        analysis = [str(content)[:800]]
    actions = parsed.get("actions")
    if not isinstance(actions, list):
        actions = []
    result = {
        "headline": str(parsed.get("headline") or status)[:40],
        "paragraphs": [str(item).strip()[:500] for item in analysis if str(item).strip()][:4],
        "actions": [str(item).strip()[:200] for item in actions if str(item).strip()][:5],
    }
    if not result["paragraphs"]:
        result["paragraphs"] = ["这一路已经出现烟雾迹象，请对照画面确认后再决定要不要派人。"]
    if not result["actions"]:
        result["actions"] = ["对照圈出的位置，确认是烟、云，还是尘和光斑。", "同一处还在就继续看，散了就记一笔后离开。"]
    with judge_lock:
        judge_cache[signature] = result
    return result


def media_path(media):
    if not isinstance(media, dict):
        return None
    kind = str(media.get("kind") or "")
    name = str(media.get("filename") or "")
    if kind == "demo":
        return demo_file(name)
    if kind == "mine":
        return session_file(str(media.get("sessionId") or ""), name)
    return None


def image_data_url(path):
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((1280, 1280))
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=80)
    except (OSError, UnidentifiedImageError):
        return ""
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def ask_report(body):
    if not isinstance(body, dict):
        raise ValueError("询问内容不对。")
    question = str(body.get("question") or "").strip()
    if not question:
        raise ValueError("先写一个问题。")
    question = question[:200]
    history = []
    for item in body.get("history") if isinstance(body.get("history"), list) else []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()[:500]
        if not text:
            continue
        history.append({
            "role": "assistant" if item.get("role") == "assistant" else "user",
            "content": text,
        })
    history = history[-6:]
    if history and history[-1]["role"] == "user":
        history = history[:-1]
    site = str(body.get("site") or "瞭望点").strip()[:40] or "瞭望点"
    code = str(body.get("code") or "").strip()[:16]
    status = str(body.get("status") or "暂无异常").strip()[:20]
    summary = str(body.get("summary") or "").strip()[:240]
    try:
        elev = int(body.get("elev") or 0)
    except (TypeError, ValueError):
        elev = 0
    weather = "，".join(
        part for part in (
            str(body.get("temp") or "").strip(),
            f"湿度 {body.get('humidity')}" if body.get("humidity") else "",
            f"{body.get('dir') or ''} {body.get('wind') or ''}".strip(),
        ) if part
    )
    facts = "\n".join(
        part for part in (
            f"点位：{site} {code}".strip(),
            f"海拔：约 {elev} 米",
            f"当前状态：{status}",
            f"工单说明：{summary}" if summary else "",
            f"现场天气：{weather}" if weather else "现场天气：暂未接通",
        ) if part
    )
    picture = ""
    path = media_path(body.get("media"))
    if path:
        picture = image_data_url(path)
    key = read_env_value("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("还没有 DeepSeek 密钥。写到 .env 的 DEEPSEEK_API_KEY 后再询问。")
    content = [{"type": "text", "text": f"工人的问题：{question}\n\n{facts}"}]
    if picture:
        content.append({"type": "image_url", "image_url": {"url": picture, "detail": "high"}})
    messages = [
        {
            "role": "system",
            "content": (
                "你是林火瞭望的现场助手。工人正在看固定相机传来的这一张画面，图片已经附在问题里。"
                "根据画面和给出的事实回答，用中文，直接说现场该怎么看、怎么做，控制在三段以内。"
                "沿用给出的当前状态，不要改成别的结论，也不要写把握、置信度、百分比或分数。"
                "不要编造明火、过火面积、被困人员，不要提到回放、历史录像或数据集。"
                "天气是暂未接通时，不要写气温、湿度和风。看不清就说看不清，并告诉工人核对烟、云、尘或光斑。"
                "不要把这些要求复述给工人。"
            ),
        },
        *history,
        {"role": "user", "content": content},
    ]

    def complete_ask():
        payload = json.dumps({
            "model": "deepseek-flash",
            "temperature": 0.4,
            "max_tokens": 4096,
            "messages": messages,
        }, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            "https://api.deepseek.com/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = json.loads(response.read().decode("utf-8"))
        message = raw["choices"][0]["message"].get("content")
        if isinstance(message, list):
            message = "".join(str(part.get("text") or "") if isinstance(part, dict) else str(part) for part in message)
        return str(message or "").strip()

    answer = ""
    try:
        for _ in range(2):
            answer = complete_ask()
            if answer:
                break
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise ValueError("DeepSeek 密钥没有通过。请核对 .env 里的 DEEPSEEK_API_KEY。") from None
        raise ValueError("询问没有完成，请再试一次。") from None
    except (urllib.error.URLError, TimeoutError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print("询问失败", type(exc).__name__, flush=True)
        raise ValueError("询问没有完成，请再试一次。") from None
    if not answer:
        raise ValueError("询问没有完成，请再试一次。")
    return {"answer": answer[:1200]}


def talk_report(body):
    if not isinstance(body, dict):
        raise ValueError("询问内容不对。")
    question = str(body.get("question") or "").strip()
    if not question:
        raise ValueError("先写一个问题。")
    question = question[:200]
    history = []
    for item in body.get("history") if isinstance(body.get("history"), list) else []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()[:500]
        if not text:
            continue
        history.append({
            "role": "assistant" if item.get("role") == "assistant" else "user",
            "content": text,
        })
    history = history[-6:]
    if history and history[-1]["role"] == "user":
        history = history[:-1]
    key = read_env_value("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("还没有 DeepSeek 密钥。写到 .env 的 DEEPSEEK_API_KEY 后再询问。")
    payload = json.dumps({
        "model": "deepseek-chat",
        "temperature": 0.4,
        "max_tokens": 800,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是烟雾瞭望的现场助手，陪工人对话。用中文，直接回答，一次不要超过三段。"
                    "可以谈怎么区分烟、云、尘和光斑，到了点位先核对哪里，照片怎么拍，工单怎么提交。"
                    "烟从地面或林线里长出来，边缘发毛，过一两分钟还在原处或顺着风挪。云的边更整齐，整团一起动。只闪一下的亮斑先记下来。"
                    "工单可以同时选已确认、已拍照、已清除、持续观察或其他处置，提交后才生效。已清除会进入待审核。"
                    "不要写把握、置信度、百分比或分数。不要编造明火、过火面积、被困人员。不要提到回放、历史录像或数据集。"
                    "看不清就说看不清。不要把这些要求复述给工人。"
                ),
            },
            *history,
            {"role": "user", "content": question},
        ],
    }, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = json.loads(response.read().decode("utf-8"))
        message = raw["choices"][0]["message"]["content"]
        if isinstance(message, list):
            message = "".join(str(part.get("text") or "") if isinstance(part, dict) else str(part) for part in message)
        answer = str(message or "").strip()
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise ValueError("DeepSeek 密钥没有通过。请核对 .env 里的 DEEPSEEK_API_KEY。") from None
        raise ValueError("询问没有完成，请再试一次。") from None
    except (urllib.error.URLError, TimeoutError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print("对话失败", type(exc).__name__, flush=True)
        raise ValueError("询问没有完成，请再试一次。") from None
    if not answer:
        raise ValueError("询问没有完成，请再试一次。")
    return {"answer": answer[:1200]}


def read_env_value(name):
    found = os.environ.get(name, "").strip()
    path = ROOT / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == name:
                found = value.strip().strip('"').strip("'")
    return found


def client_config():
    return {
        "amapKey": read_env_value("AMAP_KEY"),
        "amapSecurity": read_env_value("AMAP_SECURITY"),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "AICSeek/1"

    def do_GET(self):
        try:
            self.route_get()
        except Exception as exc:
            print("GET", self.path, exc)
            self.send_json(500, {"message": "页面没有准备好，请再试一次。"})

    def do_POST(self):
        try:
            self.route_post()
        except ValueError as exc:
            self.send_json(400, {"message": str(exc)})
        except Exception as exc:
            print("POST", self.path, exc)
            self.send_json(500, {"message": "没有完成，请再试一次。"})

    def do_DELETE(self):
        try:
            self.route_delete()
        except ValueError as exc:
            self.send_json(400, {"message": str(exc)})
        except Exception as exc:
            print("DELETE", self.path, exc)
            self.send_json(500, {"message": "没有完成，请再试一次。"})

    def route_get(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)
        if path in {"/", "/index.html", "/field"}:
            return self.send_file(STATIC / "index.html", "text/html; charset=utf-8", cache=False)
        if path == "/favicon.ico":
            return self.send_file(STATIC / "favicon.svg", "image/svg+xml", cache=True)
        if path.startswith("/static/"):
            target = safe_static(path[len("/static/"):])
            if target is None:
                return self.send_json(404, {"message": "没有这个文件。"})
            mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if target.suffix == ".svg":
                mime = "image/svg+xml"
            if mime.startswith("text/") or mime == "application/javascript":
                mime += "; charset=utf-8"
            return self.send_file(target, mime, cache=False)
        if path == "/api/config":
            return self.send_json(200, client_config())
        if path == "/api/board":
            return self.send_json(200, load_board())
        if path == "/api/live":
            try:
                lat = float(query.get("lat", ["34.53"])[0])
                lng = float(query.get("lng", ["-119.98"])[0])
            except ValueError:
                return self.send_json(400, {"message": "坐标不对。"})
            if not (-90 <= lat <= 90 and -180 <= lng <= 180):
                return self.send_json(400, {"message": "坐标不对。"})
            return self.send_json(200, live_report(lat, lng))
        if path == "/api/ready":
            ready = MODEL.is_file()
            return self.send_json(200, {
                "model": ready,
                "hint": "" if ready else "自动检查需要把模型放到 models/v0.6.0/model.zip，不用解压。",
                "model_url": "https://huggingface.co/pyronear/temporal-model/tree/v0.6.0",
            })
        if path == "/api/demo":
            return self.send_json(200, get_demo())
        if path == "/api/watch":
            return self.send_json(200, watch_snapshot())
        if path.startswith("/api/media/demo/"):
            target = demo_file(Path(path).name)
            if target is None:
                return self.send_json(404, {"message": "没有这张照片。"})
            return self.send_image(target, query)
        if path.startswith("/api/media/chat/"):
            target = chat_path(Path(path).name)
            if target is None or not target.is_file():
                return self.send_json(404, {"message": "没有这张照片。"})
            return self.send_image(target, query)
        if path.startswith("/api/media/mine/"):
            parts = [part for part in path.split("/") if part]
            if len(parts) != 5:
                return self.send_json(404, {"message": "没有这张照片。"})
            target = session_file(parts[3], parts[4])
            if target is None:
                return self.send_json(404, {"message": "没有这张照片。"})
            return self.send_image(target, query)
        if path.startswith("/api/sessions/"):
            identifier = Path(path).name
            with sessions_lock:
                session = sessions.get(identifier)
            if session is None:
                return self.send_json(404, {"message": "这组照片已经不在了，请重新放入。"})
            return self.send_json(200, session_payload(session))
        self.send_json(404, {"message": "没有这个地址。"})

    def do_PUT(self):
        try:
            self.route_put()
        except ValueError as exc:
            self.send_json(400, {"message": str(exc)})
        except Exception as exc:
            print("PUT", self.path, exc)
            self.send_json(500, {"message": "没有完成，请再试一次。"})

    def read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > 2_000_000:
            raise ValueError("内容太大。")
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def route_put(self):
        path = unquote(urlparse(self.path).path)
        if path == "/api/board":
            board, saved = store_board(self.read_json())
            return self.send_json(200 if saved else 409, board)
        self.send_json(404, {"message": "没有这个地址。"})

    def route_post(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)
        if path == "/api/board/reset":
            return self.send_json(200, reset_board())
        if path == "/api/judge":
            return self.send_json(200, judge_report(self.read_json()))
        if path == "/api/ask":
            return self.send_json(200, ask_report(self.read_json()))
        if path == "/api/talk":
            return self.send_json(200, talk_report(self.read_json()))
        if path == "/api/chat-photos":
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_UPLOAD:
                return self.send_json(400, {"message": "这张照片太大了，或是空的。单张请小于 20MB。"})
            data = self.rfile.read(length)
            name = Path(unquote(query.get("name", ["photo.jpg"])[0])).name
            return self.send_json(200, {"filename": save_chat_photo(name, data)})
        if path == "/api/sessions":
            session = new_session()
            return self.send_json(200, {"id": session["id"]})
        if path.startswith("/api/sessions/") and path.endswith("/photos"):
            identifier = path.split("/")[3]
            with sessions_lock:
                session = sessions.get(identifier)
            if session is None:
                return self.send_json(404, {"message": "这组照片已经不在了，请重新放入。"})
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_UPLOAD:
                return self.send_json(400, {"message": "这张照片太大了，或是空的。单张请小于 20MB。"})
            data = self.rfile.read(length)
            name = Path(unquote(query.get("name", [""])[0])).name
            save_upload(session, name, data)
            return self.send_json(200, {"count": len(session["files"])})
        if path.startswith("/api/sessions/") and path.endswith("/run"):
            identifier = path.split("/")[3]
            with sessions_lock:
                session = sessions.get(identifier)
            if session is None:
                return self.send_json(404, {"message": "这组照片已经不在了，请重新放入。"})
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            start_check(session, body.get("device") or "cuda")
            return self.send_json(200, {"status": "running"})
        self.send_json(404, {"message": "没有这个地址。"})

    def route_delete(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)
        if path.startswith("/api/sessions/") and path.endswith("/photos"):
            identifier = path.split("/")[3]
            with sessions_lock:
                session = sessions.get(identifier)
            if session is None:
                return self.send_json(404, {"message": "这组照片已经不在了，请重新放入。"})
            delete_upload(session, unquote(query.get("name", [""])[0]))
            return self.send_json(200, {"count": len(session["files"])})
        self.send_json(404, {"message": "没有这个地址。"})

    def send_image(self, source, query):
        if all(key in query for key in ("x", "y", "w", "h")):
            box = tuple(float(query[key][0]) for key in ("x", "y", "w", "h"))
            token = hashlib.sha1(f"crop-{source}-{source.stat().st_mtime_ns}-{box}".encode()).hexdigest()
            data = cached_jpeg(source, token, lambda src, dest: render_crop(src, dest, box))
            return self.send_bytes(data, "image/jpeg", cache=True)
        size = int(float(query.get("size", ["1440"])[0]))
        if size not in SIZES:
            size = 1440
        token = hashlib.sha1(f"s{size}-{source}-{source.stat().st_mtime_ns}".encode()).hexdigest()
        data = cached_jpeg(source, token, lambda src, dest: render_size(src, dest, size))
        self.send_bytes(data, "image/jpeg", cache=True)

    def send_file(self, path, content_type, cache):
        if not path.is_file():
            return self.send_json(404, {"message": "没有这个文件。"})
        self.send_bytes(path.read_bytes(), content_type, cache)

    def send_json(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_bytes(data, "application/json; charset=utf-8", cache=False, status=status)

    def send_bytes(self, data, content_type, cache, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=86400" if cache else "no-store")
        self.end_headers()
        self.wfile.write(data)


def safe_static(name):
    root = STATIC.resolve()
    path = (root / name).resolve()
    if path != root and root not in path.parents:
        return None
    return path if path.is_file() else None


def check_shapes():
    demo = build_demo()
    names = {frame["filename"] for frame in demo["frames"]}
    files = {path.name for path in DATA.glob("*.jpg")}
    blank = all(frame["standard"]["pending"] and not frame["standard"]["shapes"] for frame in demo["frames"])
    print("frames", len(demo["frames"]), "blank", blank)
    return names == files and blank


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        raise SystemExit(0 if check_shapes() else 1)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    cleanup_sessions()
    migrate_board()
    Image.MAX_IMAGE_PIXELS = 40_000_000
    threading.Thread(target=run_watch, daemon=True).start()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"调度中心  http://{HOST}:{PORT}/")
    print(f"工人端    http://{HOST}:{PORT}/field")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
