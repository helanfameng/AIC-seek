"""Inspect each causal prefix without changing the packaged model settings."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess

from PIL import Image
import torch
from temporal_model.core.model import BboxTubeTemporalModel
from temporal_model.core.inference import pad_frames_symmetrically, pad_frames_uniform
from temporal_model.core.tubes import build_tubes
from temporal_model.core.crop import expand_bbox, norm_bbox_to_pixel_square


def git_output(*args):
    result = subprocess.run(['git', *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else 'unavailable'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    source = Path(args.sequence).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit('Output directory is not empty. Choose a new directory to preserve results.')
    paths = sorted(p for p in source.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if not paths:
        raise SystemExit('No input images.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable.')
    model = BboxTubeTemporalModel.from_package(Path(args.model), device=args.device)
    cfg = model._cfg  # Diagnostic access, tied to the pinned upstream implementation.
    if len(paths) > cfg['classifier']['max_frames']:
        raise SystemExit('Use one sequence no longer than the packaged max_frames setting.')
    frames = model.load_sequence(paths)
    report = {'code_commit': git_output('rev-parse', 'HEAD'),
              'code_status': git_output('status', '--short'),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'torch': torch.__version__, 'device': args.device,
              'config': cfg, 'frames': [p.name for p in paths], 'prefixes': []}
    summaries = []
    views = []
    for count in range(1, len(frames) + 1):
        prefix = frames[:count]
        minimum = int(cfg['infer'].get('pad_to_min_frames', 0))
        pad_fn = {'symmetric': pad_frames_symmetrically, 'uniform': pad_frames_uniform}[
            cfg['infer'].get('pad_strategy', 'symmetric')]
        effective, padded = pad_fn(prefix, min_length=minimum)
        detections = model.detect(effective)
        # Only images available at this prefix are detected. No future crops or tracks.
        supplied = {d.frame_id: d for d in detections}
        prediction = model.predict(prefix, frame_detections=supplied, compute_trigger=False)
        tc = cfg['tubes']
        raw_tubes = build_tubes(detections, iou_threshold=tc['iou_threshold'], max_misses=tc['max_misses'])
        candidates = []
        for tube in raw_tubes:
            observations = sum(e.detection is not None for e in tube.entries)
            reasons = []
            if tube.end_frame - tube.start_frame + 1 < tc['infer_min_tube_length']:
                reasons.append('short_span')
            if observations < tc['min_detected_entries']:
                reasons.append('too_few_detection_entries')
            candidates.append({'tube': asdict(tube), 'pre_merge_filter_failures': reasons})
        kept = prediction.details['tubes']['kept']
        probabilities = [t['probability'] for t in kept if t['probability'] is not None]
        current_id = prefix[-1].frame_id
        current = supplied[current_id]
        row = {'prefix_frames': count, 'last_frame': paths[count-1].name,
               'raw_boxes_current': len(current.detections), 'raw_tubes': len(raw_tubes),
               'kept_tubes': len(kept), 'is_smoke': bool(prediction.is_positive),
               'max_probability': max(probabilities) if probabilities else None,
               'padded_slots': len(padded)}
        summaries.append(row)
        report['prefixes'].append({'summary': row, 'effective_frame_ids': [f.frame_id for f in effective],
                                   'raw_detections': [asdict(d) for d in detections],
                                   'raw_tubes': candidates, 'prediction': asdict(prediction)})
        with Image.open(paths[count-1]) as im:
            width, height = im.size
        def rect(box):
            cx, cy, w, h = box
            return [(cx-w/2)*width, (cy-h/2)*height, w*width, h*height]
        shapes = [{'rect': rect([d.cx,d.cy,d.w,d.h]), 'color':'#ffdc55', 'label':f'YOLO {d.confidence:.3f}'} for d in current.detections]
        for tube in kept:
            for entry in tube['entries']:
                if effective[entry['frame_idx']].frame_id != current_id or entry['bbox'] is None:
                    continue
                shapes.append({'rect':rect(entry['bbox']), 'color':'#60ffa3', 'label':'kept'})
                window = tube.get('stabilized_window') or entry['bbox']
                expanded = expand_bbox(*window, cfg['model_input']['context_factor'])
                x0,y0,x1,y1 = norm_bbox_to_pixel_square(*expanded,width,height)
                shapes.append({'rect':[x0,y0,x1-x0,y1-y0], 'color':'#65baff', 'label':'classifier crop'})
        image_uri = Path(os.path.relpath(paths[count-1], output)).as_posix()
        views.append({'summary':row, 'image':image_uri, 'width':width,'height':height,'shapes':shapes})
        print(f"Prefix {count}/{len(frames)}: boxes={row['raw_boxes_current']} kept={len(kept)} smoke={prediction.is_positive} p={row['max_probability']}", flush=True)
    report['first_positive_prefix_frames'] = next((r['prefix_frames'] for r in summaries if r['is_smoke']), None)
    output.mkdir(parents=True, exist_ok=True)
    (output/'diagnostics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    with (output/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    page = '''<!doctype html><meta charset="utf-8"><title>逐步输入诊断</title><style>body{background:#13202c;color:#eee;font:17px Arial;margin:24px}canvas{width:100%;max-width:1500px}p{line-height:1.7}input{width:60%}</style><h1>截至当前帧的原版模型判断</h1><p>黄：原始YOLO框（阈值与NMS之后）；绿：保留轨迹框；蓝：实际分类裁剪范围。重合框可能遮盖颜色。每个位置重新构建截至该帧的轨迹，未使用后续帧。</p><input id="slider" type="range" min="0" value="0"><p id="info"></p><canvas id="canvas"></canvas><p>不足6张时保留官方重复帧填充规则；重复帧不代表独立观测。原始框数量包括所有位置的候选，不自动等于正确烟雾框。数据：HPWREN / UCSD FIgLib。</p><script>const rows=__VIEWS__;const slider=document.getElementById('slider'),canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');slider.max=rows.length-1;let generation=0;function draw(){const token=++generation,r=rows[Number(slider.value)],s=r.summary;document.getElementById('info').textContent=`前${s.prefix_frames}张 | 当前YOLO框 ${s.raw_boxes_current} | 保留轨迹 ${s.kept_tubes} | 报警 ${s.is_smoke} | 概率 ${s.max_probability} | 填充 ${s.padded_slots}`;const im=new Image();im.onload=()=>{if(token!==generation)return;canvas.width=r.width;canvas.height=r.height;ctx.drawImage(im,0,0);ctx.lineWidth=4;ctx.font='28px Arial';for(const b of r.shapes){ctx.strokeStyle=b.color;ctx.fillStyle=b.color;ctx.strokeRect(...b.rect);ctx.fillText(b.label,b.rect[0],Math.max(30,b.rect[1]-10));}};im.src=r.image;}slider.oninput=draw;draw();</script>'''
    (output/'visualization.html').write_text(page.replace('__VIEWS__',json.dumps(views,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    (output/'README.md').write_text('# 原版候选与逐步输入诊断\n\n由diagnose_sequence.py生成；不改变模型阈值、分辨率或填充规则。\n\n查看summary.csv和visualization.html；完整配置、原始框、合并前轨迹及筛选条件、每次预测位于diagnostics.json。pre_merge_filter_failures只描述初次长度/观测数筛选，合并后轨迹ID不能简单一一对应。\n\n首次阳性前缀帧数：'+str(report['first_positive_prefix_frames'])+'。本数据每60秒一帧，首帧是官方首次可见烟羽；不是实际点火时间。尚未比较降阈值或高分辨率，也不能据单个样例推断准确率。\n',encoding='utf-8')
    print('Saved:',output)


if __name__ == '__main__':
    main()
