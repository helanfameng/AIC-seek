"""One shared annotation file and a local review UI. Empty VIA boxes stay unknown."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
import shutil
import time
from urllib.parse import parse_qs, urlparse

from common import DATA, EXPERIMENT, load_manifest, now, read_json
from review_assist import annotation, apply_batch, group_candidates, labels_version, load_candidates, save_labels


def prepare():
    manifest = load_manifest()
    path = DATA / "labels.json"
    labels = read_json(path) if path.exists() else {"schema_version": 1, "sequences": {}}
    changed = not path.exists()
    for seq in manifest["sequences"]:
        changed = changed or seq["id"] not in labels["sequences"]
        labels["sequences"].setdefault(seq["id"], {"normal_status": "unreviewed",
            "candidate_review_complete": {}, "human_visible_timestamp": None,
            "frames": {}, "candidates": {}})
    if changed:
        save_labels(path, labels)
    print("Shared annotations:", path)


def import_via(path):
    prepare()
    value = read_json(path)
    metadata = value.get("_via_img_metadata", value)
    manifest = load_manifest()
    labels = read_json(DATA / "labels.json")
    rows = [(seq, f) for seq in manifest["sequences"] for f in seq["frames"]]
    changed = 0
    for image in metadata.values():
        if not isinstance(image, dict) or "filename" not in image:
            continue
        filename = image["filename"].replace("\\", "/")
        found = [(s, f) for s, f in rows if f["file"] == filename or f["file"].endswith("/" + filename)]
        if not found:
            found = [(s, f) for s, f in rows if Path(f["file"]).name == Path(filename).name]
        if len(found) != 1:
            raise ValueError(f"VIA filename missing/ambiguous; use sequence-relative paths: {filename}")
        seq, frame = found[0]
        boxes = []
        regions = image.get("regions", [])
        if isinstance(regions, dict):
            regions = list(regions.values())
        for region in regions:
            shape = region.get("shape_attributes", {})
            if shape.get("name") != "rect":
                continue
            x, y, w, h = (shape[k] for k in ("x", "y", "width", "height"))
            boxes.append([x / frame["width"], y / frame["height"],
                          (x + w) / frame["width"], (y + h) / frame["height"]])
        attrs = image.get("file_attributes", {})
        status = attrs.get("status", attrs.get("visibility"))
        if status not in ("visible", "absent", "uncertain"):
            status = "visible" if boxes else None
        # Empty boxes never mean absent by themselves, including later omitted boxes.
        if status is None:
            continue
        state = labels["sequences"][seq["id"]]
        state["frames"][Path(frame["file"]).stem] = {"status": status, "smoke_boxes": boxes,
                                                    "source": str(Path(path).name), "updated_at": now()}
        changed += 1
    save_labels(DATA / "labels.json", labels)
    print(f"Imported {changed} explicitly labeled frames; empty VIA entries remained unknown")


def serve(args):
    prepare()
    manifest = load_manifest()
    by_id = {s["id"]: s for s in manifest["sequences"] if
             args.review_split == "all" or s["split"] == args.review_split or
             (args.review_split == "develop" and s["split"] in ("train", "val"))}
    allowed = {f["file"] for s in by_id.values() for f in s["frames"]}
    folder = EXPERIMENT / "outputs" / args.run_id / args.setting
    candidates = load_candidates(folder)
    label_path = DATA / "labels.json"
    backup_path = None

    def suggestions():
        path = folder / "review_suggestions.json"
        return read_json(path) if path.exists() else {"candidates": {}, "counts": {}}

    def sequence_data(seq_id):
        labels = read_json(label_path)
        seq = by_id[seq_id]
        state = labels["sequences"][seq_id]
        proposed = suggestions().get("candidates", {})
        rows = candidates.get(seq_id, [])
        groups = group_candidates(rows, state, proposed)
        return dict(seq, labels=state, candidates=[dict(row, reviewed=annotation(row, state)[0],
            reviewed_label=annotation(row, state)[1], reviewed_target=annotation(row, state)[2],
            suggestion=proposed.get(row["id"], {"label": None, "reason": "尚未生成预标注"})) for row in rows],
            groups=groups, version=labels_version(labels), setting=args.setting, run_id=args.run_id)

    def thumbnail(row):
        from PIL import Image, ImageDraw, ImageOps
        with Image.open(DATA / row["frame_file"]) as original:
            image = original.convert("RGB")
        width, height = image.size
        a, b, c, d = row["crop_box"]
        box = (int(a * width), int(b * height), int(c * width), int(d * height))
        pad_x, pad_y = max(60, (box[2] - box[0]) * .6), max(45, (box[3] - box[1]) * .6)
        context = (max(0, int(box[0] - pad_x)), max(0, int(box[1] - pad_y)),
                   min(width, int(box[2] + pad_x)), min(height, int(box[3] + pad_y)))
        image = image.crop(context)
        ImageDraw.Draw(image).rectangle((box[0] - context[0], box[1] - context[1],
                                        box[2] - context[0], box[3] - context[1]),
                                       outline=(255, 90, 60), width=max(2, image.width // 200))
        image = ImageOps.contain(image, (480, 270))
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=85)
        return output.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *values):
            pass

        def reply(self, value, status=200, content_type="application/json"):
            body = json.dumps(value, ensure_ascii=False).encode() if content_type == "application/json" else value
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    return self.reply(Path(__file__).with_name("review.html").read_bytes(), content_type="text/html")
                if parsed.path == "/api/sequences":
                    labels = read_json(label_path)
                    result = []
                    for seq in by_id.values():
                        state = labels["sequences"][seq["id"]]
                        rows = candidates.get(seq["id"], [])
                        result.append({k: seq[k] for k in ("id", "split", "source")} | {
                            "normal_status": state.get("normal_status"), "total": len(rows),
                            "pending": sum(not annotation(row, state)[0] for row in rows)})
                    return self.reply(result)
                if parsed.path == "/api/sequence":
                    return self.reply(sequence_data(query["id"][0]))
                if parsed.path == "/api/suggestions":
                    value = suggestions()
                    return self.reply({k: v for k, v in value.items() if k != "candidates"})
                if parsed.path == "/thumbnail":
                    seq_id, candidate_id = query["sequence"][0], query["id"][0]
                    if seq_id not in by_id:
                        raise ValueError("Sequence is outside the current review split")
                    row = next(r for r in candidates.get(seq_id, []) if r["id"] == candidate_id)
                    return self.reply(thumbnail(row), content_type="image/jpeg")
                if parsed.path == "/image":
                    name = query["file"][0]
                    if name not in allowed:
                        raise ValueError("Image is outside the manifest")
                    return self.reply((DATA / name).read_bytes(), content_type="image/jpeg")
                self.reply({"error": "not found"}, 404)
            except Exception as error:
                self.reply({"error": str(error)}, 400)

        def do_POST(self):
            nonlocal backup_path
            try:
                if self.path == "/api/prelabel":
                    from review_assist import prelabel
                    result = prelabel(args)
                    return self.reply({"saved": True, "counts": result["counts"], "formal_labels_changed": False})
                if self.path != "/api/update":
                    raise ValueError("Unknown endpoint")
                size = int(self.headers.get("Content-Length", "0"))
                if size > 1024 * 1024:
                    raise ValueError("Request too large")
                item = json.loads(self.rfile.read(size))
                seq = by_id[item["sequence"]]
                labels = read_json(label_path)
                if item.get("version") and item["version"] != labels_version(labels):
                    return self.reply({"error": "其他页面更新了标注。本页已刷新，请重新操作；已有记录未丢失。"}, 409)
                state = labels["sequences"][seq["id"]]
                action = item["action"]
                rows = candidates.get(seq["id"], [])
                result = {}
                if action == "frame":
                    if item["frame_file"] not in {f["file"] for f in seq["frames"]}:
                        raise ValueError("Frame not in sequence")
                    if item["status"] not in ("visible", "absent", "uncertain"):
                        raise ValueError("Unknown visibility")
                    state["frames"][Path(item["frame_file"]).stem] = {
                        "status": item["status"], "smoke_boxes": item["smoke_boxes"], "updated_at": now()}
                    if seq["source"] == "hpwren" and item["status"] == "visible":
                        state["normal_status"] = "contains_smoke"
                elif action == "candidate":
                    result = apply_batch(state, rows, [item], overwrite=True, source="human_single")
                elif action == "batch":
                    result = apply_batch(state, rows, item["items"], overwrite=bool(item.get("overwrite")))
                elif action == "accept_suggestions":
                    proposed = suggestions().get("candidates", {})
                    groups = {g["id"]: g for g in group_candidates(rows, state, proposed)}
                    chosen = item.get("groups", [])
                    if not chosen or any(g not in groups for g in chosen) or len(chosen) != len(set(chosen)):
                        raise ValueError("请先浏览并选择有效分组")
                    selected = {candidate_id for group in chosen for candidate_id in groups[group]["ids"]}
                    pending = [row for row in rows if row["id"] in selected and not annotation(row, state)[0]]
                    if any(row["id"] not in proposed for row in pending):
                        raise ValueError("部分候选尚无预标注，请先生成预标注，或手工整组确认")
                    items = [{"id": row["id"], "label": proposed[row["id"]]["label"],
                              "matches_target": seq["source"] == "figlib" and bool(item.get("matches_target"))}
                             for row in pending]
                    if items:
                        result = apply_batch(state, rows, items, source="human_accepted_suggestion")
                    if item.get("complete"):
                        missing = sum(not annotation(row, state)[0] for row in rows)
                        if missing:
                            raise ValueError(f"还有{missing}个候选未复核，本次未保存；请继续查看")
                        state.setdefault("candidate_review_complete", {})[args.setting] = True
                elif action == "normal" and seq["source"] == "hpwren":
                    if any(f.get("status") == "visible" for f in state["frames"].values()) or \
                       any(c.get("label") == 1 for c in state["candidates"].values()):
                        raise ValueError("This clip contains explicitly visible smoke; exclude it from normal footage")
                    state["normal_status"] = "confirmed_normal"
                elif action == "reject_normal" and seq["source"] == "hpwren":
                    state["normal_status"] = "excluded"
                    state["exclusion_reason"] = item["reason"]
                elif action == "complete":
                    if state.get("normal_status") not in ("excluded", "contains_smoke"):
                        missing = sum(not annotation(row, state)[0] for row in rows)
                        if missing:
                            raise ValueError(f"还有{missing}个候选未复核，请先完成分组确认；无法判断的标为不确定")
                    state.setdefault("candidate_review_complete", {})[args.setting] = True
                elif action == "visible_time":
                    if seq["source"] != "figlib" or not any(f["timestamp"] == item["timestamp"] for f in seq["frames"]):
                        raise ValueError("人工首次可见仅用于火情序列，请选择有效帧")
                    state["human_visible_timestamp"] = int(item["timestamp"])
                else:
                    raise ValueError("Unknown review action")
                if backup_path is None:
                    backup_path = DATA / f"labels.before_session_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}.json"
                    shutil.copy2(label_path, backup_path)
                save_labels(label_path, labels)
                self.reply({"saved": True, "version": labels_version(labels), **result})
            except Exception as error:
                self.reply({"error": str(error)}, 400)

    print(f"Review UI: http://127.0.0.1:{args.port} (Ctrl+C to stop)", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
