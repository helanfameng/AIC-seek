"""Reuse candidate review UI and add explicit sequence eligibility review."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import time
from urllib.parse import parse_qs, urlparse

import numpy as np

import common7 as c
from data7 import annotation
from learning7 import current_features
from legacy7 import bind


def prelabel(args):
    """Train-only appearance suggestions; no model scores or automatic truth."""
    rows = c.load_rows(args)
    annotations = c.labels()
    cfg = c.protocol()
    buckets = defaultdict(list)
    for row in rows:
        if row["split"] != "train" or not row["feature_file"]:
            continue
        state = annotations["sequences"][row["sequence"]]
        if row["source"] == "hpwren" and state["normal_status"] != "confirmed_normal":
            continue
        label = annotation(row, annotations)[1]
        if label in (0, 1):
            buckets[(row["sequence"], label)].append(row)
    references = {0: [], 1: []}
    for (_, label), items in buckets.items():
        for index in np.linspace(0, len(items)-1, min(40, len(items)), dtype=int):
            row = items[int(index)]
            vector = current_features(row, cfg["feature_dimension"])
            vector /= max(float(np.linalg.norm(vector)), 1e-8)
            references[label].append((row["id"], vector))
    banks = {label: np.stack([v for _, v in values]) if values else None
             for label, values in references.items()}
    splits = ("train", "val") if args.review_split == "develop" else (
        ("train", "val", "test") if args.review_split == "all" else (args.review_split,))
    proposed = {}
    for row in rows:
        if row["split"] not in splits or annotation(row, annotations)[0]:
            continue
        value = {"label": None, "reason": "相似证据不足，请看原图确认", "method": "train_only_feature_retrieval"}
        if row["feature_file"] and all(banks[y] is not None for y in (0, 1)):
            vector = current_features(row, cfg["feature_dimension"])
            vector /= max(float(np.linalg.norm(vector)), 1e-8)
            similarities = {y: banks[y] @ vector for y in (0, 1)}
            best = {y: float(similarities[y].max()) for y in (0, 1)}
            label = int(best[1] > best[0])
            if best[label] >= 0.92 and best[label] - best[1-label] >= 0.04:
                value.update(label=label, reason="与已确认训练样本相似，仅为待确认建议",
                             reference=references[label][int(similarities[label].argmax())][0])
        proposed[row["id"]] = value
    path = c.folder(args) / "review_suggestions.json"
    if path.exists():
        previous = c.read_json(path).get("candidates", {})
        ids = {r["id"] for r in rows if r["split"] in splits}
        proposed = {k: v for k, v in previous.items() if k not in ids} | proposed
    result = {"created_at": c.now(), "purpose": "unconfirmed_annotation_assistance_only",
              "reference_split": "train_only", "reference_fingerprint": c.fingerprint("train"),
              "seed_counts": {str(k): len(v) for k, v in references.items()},
              "counts": {"smoke": sum(v["label"] == 1 for v in proposed.values()),
                         "background": sum(v["label"] == 0 for v in proposed.values()),
                         "uncertain": sum(v["label"] is None for v in proposed.values())},
              "candidates": proposed}
    c.write_json(path, result)
    print("Unconfirmed review suggestions:", result["counts"], flush=True)
    return result


def update_eligibility(item):
    path = c.DATA / "eligibility.json"
    value = c.read_json(path)
    if item.get("version") != c.object_digest(value):
        raise ValueError("资格记录已在另一页面更新，请刷新后重试")
    seq_id = item.get("sequence")
    if seq_id not in value["sequences"]:
        raise ValueError("未知测试序列")
    status = item.get("status")
    if status not in ("eligible", "uncertain", "excluded"):
        raise ValueError("未知资格状态")
    reason = str(item.get("reason", "")).strip()
    if status != "eligible" and not reason:
        raise ValueError("不确定或排除时需要填写原因")
    if not list(c.DATA.glob("eligibility.before_session_*.json")):
        shutil.copy2(path, c.DATA / f"eligibility.before_session_{time.time_ns()}.json")
    value["sequences"][seq_id] = {"status": status, "reason": reason,
                                 "source": "human_eligibility_ui", "updated_at": c.now()}
    c.write_json(path, value)
    return {"saved": True, "version": c.object_digest(value)}


def serve(args):
    modules = bind()
    review, assist = modules["review"], modules["review_assist"]
    original_server, original_prelabel = review.HTTPServer, assist.prelabel
    assist.prelabel = prelabel

    class Server(original_server):
        def __init__(self, address, original_handler):
            class Handler(original_handler):
                def do_GET(self):
                    parsed = urlparse(self.path)
                    try:
                        if parsed.path == "/":
                            html = Path(review.__file__).with_name("review.html").read_text(encoding="utf-8")
                            html = html.replace("实验6", "实验7").replace("exp006.", "exp007.")
                            html = html.replace("<main>", '<main><p><a style="color:#8bd9d1" href="/eligibility" target="_blank">打开10个主对照事件的资格审核</a> · 候选真假与序列资格分别保存。</p>', 1)
                            return self.reply(html.encode(), content_type="text/html")
                        if parsed.path == "/eligibility":
                            return self.reply(Path(__file__).with_name("eligibility.html").read_bytes(), content_type="text/html")
                        if parsed.path == "/api/eligibility":
                            value = c.read_json(c.DATA / "eligibility.json")
                            return self.reply({"version": c.object_digest(value), "sequences": [
                                {"id": s["id"], "frames": s["frames"], "official_onset": s["official_onset"],
                                 "review": value["sequences"].get(s["id"], {})}
                                for s in c.manifest()["sequences"] if s["source"] == "figlib" and s["split"] == "test"]})
                    except Exception as error:
                        return self.reply({"error": str(error)}, 400)
                    return super().do_GET()

                def do_POST(self):
                    if self.path == "/api/eligibility":
                        try:
                            size = int(self.headers.get("Content-Length", "0"))
                            if not 0 < size <= 16384:
                                raise ValueError("无效请求大小")
                            return self.reply(update_eligibility(json.loads(self.rfile.read(size))))
                        except Exception as error:
                            return self.reply({"error": str(error)}, 400)
                    return super().do_POST()
            super().__init__(address, Handler)

    review.HTTPServer = Server
    try:
        review.serve(args)
    finally:
        review.HTTPServer, assist.prelabel = original_server, original_prelabel
