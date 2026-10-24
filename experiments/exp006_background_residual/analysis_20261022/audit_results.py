"""Read-only audit; no training, label changes or test threshold selection.

Run with Python 3.12 and the experiment numpy/torch/matplotlib environment.
All derived output is written beside this script.
"""
from __future__ import annotations
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

OUT = Path(__file__).resolve().parent
EXP = OUT.parent
ROOT = EXP.parents[1]
DATA = ROOT / 'data/exp006_background_residual'
RUN = EXP / 'outputs/exp006_main_20261021/baseline'
SEED = RUN / 'seed_42'
packages = ROOT.parent / 'pyronear_lab/.venv/Lib/site-packages'
if packages.exists():
    sys.path.insert(0, str(packages))
sys.dont_write_bytecode = True
os.environ['MPLCONFIGDIR'] = str(OUT / 'render_cache')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / '.cache/exp006/temp')

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def jsonl(path):
    return [json.loads(t) for t in path.read_text(encoding='utf-8').splitlines() if t.strip()]

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

manifest, labels, cfg = read(DATA/'manifest.json'), read(DATA/'labels.json'), read(EXP/'protocol.json')
thresholds, summary, run = read(SEED/'thresholds.json'), read(SEED/'summary.json'), read(RUN/'run.json')
sequences = manifest['sequences']
seq_by_id = {s['id']: s for s in sequences}
rows = jsonl(SEED/'test_predictions.jsonl')
all_rows = [r for sid in run['sequence_ids'] for r in jsonl(RUN/f'{sid}.jsonl')]
variants = list(summary['variants'])
by_seq = collections.defaultdict(list)
for row in rows:
    by_seq[row['sequence']].append(row)

def normal_metrics(seq, variant, threshold):
    """Independent integration; include frames with zero candidates."""
    state = labels['sequences'][seq['id']]
    if state['normal_status'] != 'confirmed_normal':
        return None
    scores = collections.defaultdict(float)
    for r in by_seq[seq['id']]:
        scores[r['timestamp']] = max(scores[r['timestamp']], r['scores'][variant])
    total, positive, segments = 0, 0, []
    for i, frame in enumerate(seq['frames']):
        status = state.get('frames', {}).get(Path(frame['file']).stem, {}).get('status')
        end = seq['frames'][i+1]['timestamp'] if i+1 < len(seq['frames']) else seq['end_utc_exclusive']
        duration = end-frame['timestamp']
        if status in ('visible', 'uncertain') or not 0 < duration <= cfg['maximum_history_gap_seconds']:
            continue
        total += duration
        if scores[frame['timestamp']] < threshold:
            continue
        positive += duration
        if not segments or frame['timestamp']-segments[-1]['last'] > cfg['alarm_release_minutes']*60:
            segments.append(dict(first=frame['timestamp'], last=frame['timestamp'], positive_seconds=duration, frames=1))
        else:
            segments[-1]['last'] = frame['timestamp']
            segments[-1]['positive_seconds'] += duration
            segments[-1]['frames'] += 1
    return dict(sequence=seq['id'], hours=total/3600, positive_minutes=positive/60,
                episodes=len(segments), positive_fraction=positive/total, segments=segments)

def is_target(row):
    a = labels['sequences'][row['sequence']]['candidates'].get(row['id'], {})
    return a.get('label') == 1 and bool(a.get('matches_target'))

events, normal, recomputed = [], {}, {}
for variant in variants:
    th = thresholds['variants'][variant]['threshold']
    normal[variant] = [normal_metrics(s, variant, th) for s in sequences if s['split']=='test' and s['source']=='hpwren']
    ns = normal[variant]
    hours = sum(s['hours'] for s in ns)
    minutes = sum(s['positive_minutes'] for s in ns)
    episodes = sum(s['episodes'] for s in ns)
    delays = []
    for seq in sequences:
        if seq['split']!='test' or seq['source']!='figlib':
            continue
        windows = collections.defaultdict(list)
        for r in by_seq[seq['id']]:
            if r['timestamp'] >= seq['official_onset']:
                windows[r['timestamp']].append(r)
        correct = []
        for t, candidates in windows.items():
            winner = max(candidates, key=lambda r:r['scores'][variant])
            if is_target(winner) and winner['scores'][variant] >= th:
                correct.append(t)
        delay = (min(correct)-seq['official_onset'])/60 if correct else None
        if delay is not None:
            delays.append(delay)
        ref = next(e for e in summary['variants'][variant]['fire']['per_event'] if e['sequence']==seq['id'])
        assert delay == ref['delay_minutes_official'], (variant, seq['id'])
    got = dict(detected=len(delays), missed=10-len(delays), recall5=sum(d<=5 for d in delays)/10,
               recall10=sum(d<=10 for d in delays)/10, median=statistics.median(delays), mean_detected=statistics.mean(delays),
               normal_hours=hours, episodes=episodes, false_alarms_per_hour=episodes/hours,
               positive_fraction=minutes/(60*hours), positive_minutes=minutes)
    ref = summary['variants'][variant]
    assert got['detected']==ref['fire']['detected'] and got['median']==ref['fire']['median_delay_detected_only']
    assert got['episodes']==ref['normal']['false_alarm_episodes']
    assert abs(got['positive_fraction']-ref['normal']['positive_time_fraction'])<1e-12
    assert abs(hours-ref['normal']['observed_camera_hours'])<1e-12
    recomputed[variant] = got

for seq in sequences:
    if seq['split']!='test' or seq['source']!='figlib':
        continue
    rs = by_seq[seq['id']]
    post = [r for r in rs if r['timestamp']>=seq['official_onset']]
    targets = [r for r in post if is_target(r)]
    det = read(RUN/f"{seq['id']}.detections.json")
    e = dict(sequence=seq['id'], site=seq['site'], candidate_rows=len(rs), post_onset_rows=len(post),
             post_onset_target_rows=len(targets), raw_detection_boxes=sum(len(d['detections']) for d in det),
             first_target_candidate_delay=(min(r['timestamp'] for r in targets)-seq['official_onset'])/60 if targets else None,
             delays={}, preonset_background_positive_winners={}, postonset_background_positive_winners={})
    e['candidate_identity_counts']=dict(collections.Counter(f"label={r['reviewed_label']},target={bool((r.get('target_annotation') or {}).get('matches_target'))}" for r in rs))
    if targets:
        first=min(targets,key=lambda r:r['timestamp'])
        e['first_target_diagnostic']=dict(timestamp=first['timestamp'],age_seconds=first['age_seconds'],target_scores=first['scores'],winners={})
        for v in variants:
            winner=max([r for r in rs if r['timestamp']==first['timestamp']],key=lambda r:r['scores'][v])
            e['first_target_diagnostic']['winners'][v]=dict(winner_is_target=is_target(winner),winner_score=winner['scores'][v],
                target_crosses_threshold=first['scores'][v]>=thresholds['variants'][v]['threshold'])
    for v in variants:
        e['delays'][v] = next(x['delay_minutes_official'] for x in summary['variants'][v]['fire']['per_event'] if x['sequence']==seq['id'])
        for before, key in ((True,'preonset_background_positive_winners'),(False,'postonset_background_positive_winners')):
            win = collections.defaultdict(list)
            for r in rs:
                if (r['timestamp']<seq['official_onset'])==before:
                    win[r['timestamp']].append(r)
            count = 0
            for rr in win.values():
                w = max(rr,key=lambda r:r['scores'][v])
                count += w['reviewed_label']==0 and w['scores'][v]>=thresholds['variants'][v]['threshold']
            e[key][v] = count
    events.append(e)

hashes = dict(protocol=sha(EXP/'protocol.json')==thresholds['protocol_sha256']==run['protocol_sha256'],
              selection=sha(EXP/'data_selection.json')==run['manifest_selection_sha256'],
              extractor=sha(EXP/'scripts/features.py')==run['extractor_sha256'])
for v in variants[1:]:
    hashes[v] = sha(SEED/f'{v}.pt')==thresholds['checkpoint_sha256'][v]
for split, field in (('train','train_labels_sha256'),('val','validation_labels_sha256')):
    state = {s['id']:labels['sequences'][s['id']] for s in sequences if s['split']==split}
    hashes[field] = hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest()==thresholds[field]
assert all(hashes.values()), hashes
assert run['complete'] and not run['partial']
assert len({r['id'] for r in rows})==len(rows)
assert set(r['id'] for r in rows)=={r['id'] for r in all_rows if r['split']=='test'}
site_sets = {sp:{s['site'] for s in sequences if s['split']==sp} for sp in ('train','val','test')}
event_sets = {sp:{s['event_group'] for s in sequences if s['split']==sp and s.get('event_group')} for sp in site_sets}
assert all(not site_sets[a]&site_sets[b] and not event_sets[a]&event_sets[b] for a,b in (('train','val'),('train','test'),('val','test')))
history_bad = []
for r in all_rows:
    if r['feature_file']:
        t = r['history_timestamps']
        if len(t)!=8 or t[-1]!=r['timestamp'] or any(not 0<b-a<=120 for a,b in zip(t,t[1:])):
            history_bad.append(r['id'])
assert not history_bad
missing, size_mismatch = [], []
for s in sequences:
    for f in s['frames']:
        p=DATA/f['file']
        if not p.is_file(): missing.append(str(p))
        elif p.stat().st_size!=f['bytes']: size_mismatch.append(str(p))
losses = {}
for p in SEED.glob('*history.json'):
    h=read(p)
    losses[p.stem.removesuffix('_history')] = dict(epochs=len(h),best=min(h,key=lambda r:r['val']),first=h[0],last=h[-1],history=h)
audit = dict(scope='exp006_main_20261021/baseline/seed_42; frozen test thresholds; no retraining or retuning',
             raw_test_rows=len(rows), all_candidate_rows=len(all_rows), hash_checks=hashes, site_counts={k:len(v) for k,v in site_sets.items()},
             image_count=sum(len(s['frames']) for s in sequences), image_missing=missing, image_size_mismatch=size_mismatch,
             history_violations=history_bad, metrics=recomputed, events=events, normal_by_clip=normal, losses=losses,
             frozen_thresholds=thresholds, counts={})
for sp in ('train','val','test'):
    rs=[r for r in all_rows if r['split']==sp]
    audit['counts'][sp]=dict(rows=len(rs),with_features=sum(r['feature_file'] is not None for r in rs),source_rows=dict(collections.Counter(r['source'] for r in rs)))
detected = [e for e in events if e['delays']['original'] is not None]
audit['paired_improvement_minutes'] = {}
for ref in ('original','calibration_only','mean_residual'):
    d=[e['delays'][ref]-e['delays']['learned_residual'] for e in detected]
    audit['paired_improvement_minutes'][ref]=dict(per_detected_event=d,mean=statistics.mean(d),median=statistics.median(d),
                                               improved=sum(x>1e-9 for x in d),unchanged=sum(abs(x)<=1e-9 for x in d),worse=sum(x<-1e-9 for x in d))
audit['candidate_floor_hits'] = {v:sum(e['first_target_candidate_delay']==e['delays'][v] for e in detected) for v in variants}
audit['zero_alarm_poisson_95_upper_per_hour']=-math.log(.05)/recomputed['original']['normal_hours']
audit['zero_alarm_required_hours_for_0_1_95_upper']=-math.log(.05)/.1
audit['data_coverage']={}
image_splits=collections.defaultdict(set)
for s in sequences:
    for f in s['frames']: image_splits[f['sha256']].add(s['split'])
audit['cross_split_declared_image_hashes']=sum(len(v)>1 for v in image_splits.values())
assert audit['cross_split_declared_image_hashes']==0
for sp in ('train','val','test'):
    ss=[s for s in sequences if s['split']==sp]
    ns=[normal_metrics(s,'original',1.000001) for s in ss if s['source']=='hpwren' and labels['sequences'][s['id']]['normal_status']=='confirmed_normal']
    counts=collections.Counter()
    positives=set()
    for r in all_rows:
        if r['split']!=sp or not r['feature_file']: continue
        st=labels['sequences'][r['sequence']]
        a=st.get('candidates',{}).get(r['id'])
        status=st.get('frames',{}).get(Path(r['frame_file']).stem,{}).get('status')
        label=a.get('label') if a is not None and 'label' in a else (0 if status=='absent' or (status not in ('visible','uncertain') and r['source']=='hpwren' and st['normal_status']=='confirmed_normal') else None)
        counts[str(label)]+=1
        if label==1 and r['source']=='figlib': positives.add(r['sequence'])
    audit['data_coverage'][sp]=dict(confirmed_normal_clips=len(ns),normal_hours=sum(s['hours'] for s in ns),
                                    fire_events_with_positive_feature_rows=len(positives),feature_label_counts=dict(counts))
for r in rows:
    state=labels['sequences'][r['sequence']]
    assert r['target_annotation']==state['candidates'].get(r['id'])
audit['normal_alarm_evidence_age']={}
for v in variants:
    audit['normal_alarm_evidence_age'][v]=[]
    for clip in normal[v]:
        if not clip['episodes']: continue
        rr=[r for r in by_seq[clip['sequence']] if r['scores'][v]>=thresholds['variants'][v]['threshold']]
        audit['normal_alarm_evidence_age'][v].append(dict(sequence=clip['sequence'],positive_rows=len(rr),
            ages_seconds=[r['age_seconds'] for r in rr],real_evidence_last_timestamps=sorted({r['timestamp']-r['age_seconds'] for r in rr}),
            lengths=sorted({r['length'] for r in rr})))
print('Independent metric recomputation and frozen-artifact checks passed.',flush=True)

import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(EXP/'scripts'))
from learning import BackgroundPredictor
from learning import score_rows
torch.set_num_threads(4)
state=torch.load(SEED/'predictor.pt',map_location='cpu',weights_only=True)
device='cuda' if torch.cuda.is_available() else 'cpu'
model=BackgroundPredictor(state['dimension'],cfg['hidden_dim']).to(device).eval()
model.load_state_dict(state['state'])
rescored=score_rows(RUN,42,device,rows)
audit['saved_score_max_abs_difference']={v:max(abs(rescored[r['id']][v]-r['scores'][v]) for r in rows) for v in variants}
assert all(d<1e-6 for d in audit['saved_score_max_abs_difference'].values())
normal_errors=[]
with torch.inference_mode():
    for s in sequences:
        st=labels['sequences'][s['id']]
        if s['source']!='hpwren' or st['normal_status']!='confirmed_normal': continue
        bank=np.load(RUN/'features'/f"normal_{s['id']}.npz")['features']
        errors=collections.defaultdict(list)
        for end in range(5,len(s['frames'])):
            fs=s['frames'][end-5:end+1]
            if any(not 0<b['timestamp']-a['timestamp']<=120 for a,b in zip(fs,fs[1:])): continue
            if any(st.get('frames',{}).get(Path(f['file']).stem,{}).get('status') in ('visible','uncertain') for f in fs): continue
            x=torch.from_numpy(bank[end-5:end].astype(np.float32)).permute(1,0,2,3).to(device)
            target=F.normalize(torch.from_numpy(bank[end].astype(np.float32)).to(device),dim=-1)
            xn=F.normalize(x,dim=-1)
            refs=dict(last=xn[:,-1],mean=F.normalize(xn.mean(1),dim=-1),learned=model(x))
            for name,ref in refs.items(): errors[name].append(float((1-(ref*target).sum(-1)).mean().item()))
        normal_errors.append(dict(sequence=s['id'],split=s['split'],windows=len(errors['last']),cosine_loss={k:statistics.mean(v) for k,v in errors.items()}))
audit['normal_predictor_reference_diagnostic']=normal_errors
audit['normal_predictor_split_means']={sp:{name:statistics.mean(e['cosine_loss'][name] for e in normal_errors if e['split']==sp) for name in ('last','mean','learned')} for sp in ('train','val','test')}
audit['checkpoint_metadata']={}
for v in ('predictor',*variants[1:]):
    st=torch.load(SEED/f'{v}.pt',map_location='cpu',weights_only=True)
    audit['checkpoint_metadata'][v]={k:st[k] for k in ('dimension','train_windows','val_windows','train_candidates','val_candidates','class_counts','trainable_parameters') if k in st}
    if v=='calibration_only': audit['calibration_coefficients']={k:t.tolist() for k,t in st['state'].items()}
(OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Predictor comparison:',audit['normal_predictor_split_means'],flush=True)
print('Paired improvements:',audit['paired_improvement_minutes'],flush=True)
print('Candidate floor:',audit['candidate_floor_hits'],flush=True)
print('Normal alarms:',{v:[(s['sequence'],s['episodes'],s['positive_minutes']) for s in normal[v] if s['episodes']] for v in variants},flush=True)

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import font_manager
from PIL import Image
font=Path(r'C:\Windows\Fonts\msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
names=['原版','仅校准','普通历史','均值残差','学习残差']
colors=['#5d7080','#7995a5','#b3b7c1','#2c8a7c','#d26a3f']
fig,axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
fig.suptitle('实验六：提前发现与背景误报的取舍\nbaseline / seed 42；10个测试事件，9.974摄像头小时正常片段',fontsize=16)
ax=axes[0,0]
short=[e['sequence'].split('_')[1].replace('Fire','') for e in detected]
matrix=np.array([[e['delays'][v] for e in detected] for v in variants])
ax.imshow(matrix,vmin=0,vmax=14,cmap='YlOrBr',aspect='auto')
ax.set(xticks=range(len(short)),xticklabels=short,yticks=range(5),yticklabels=names,title='检出事件延迟（分钟）；另有2个事件全部漏检')
ax.tick_params(axis='x',rotation=35)
for i in range(5):
    for j in range(len(short)): ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',color='white' if matrix[i,j]>10 else '#18232c')
ax=axes[0,1]
rates=[recomputed[v]['false_alarms_per_hour'] for v in variants]
ax.bar(names,rates,color=colors)
ax.axhline(.1,color='#8b3333',linestyle='--',label='协议参考：0.1次/小时')
for i,v in enumerate(variants): ax.text(i,rates[i]+.012,f"{recomputed[v]['episodes']}次 / {rates[i]:.3f}",ha='center')
ax.set(ylim=(0,.64),ylabel='误报次数 / 摄像头小时',title='测试误报：学习残差增加至5次')
ax.legend(frameon=False,fontsize=9)
ax=axes[1,0]
pct=[recomputed[v]['positive_fraction']*100 for v in variants]
ax.bar(names,pct,color=colors)
ax.axhline(1,color='#8b3333',linestyle='--',label='协议参考：1%')
for i in range(5): ax.text(i,pct[i]+.1,f'{pct[i]:.2f}%',ha='center')
ax.set(ylim=(0,4.8),ylabel='正常时间中的超阈占比（%）',title='持续误报也应计入代价')
ax.legend(frameon=False,fontsize=9)
ax=axes[1,1]
for v,name,color in zip(('history','mean_residual','learned_residual'),('普通历史','均值残差','学习残差'),(colors[2],colors[3],colors[4])):
    h=losses[v]['history']
    ax.plot([r['epoch'] for r in h],[r['val'] for r in h],label=name,color=color,marker='.')
    best=losses[v]['best']
    ax.scatter(best['epoch'],best['val'],color=color,marker='*',s=150,edgecolor='#333333',linewidth=.5)
ax.set(xlabel='训练轮次',ylabel='验证候选加权BCE',title='验证损失；星号为保存检查点')
ax.legend(frameon=False)
fig.savefig(OUT/'comparison.png',dpi=160)
plt.close(fig)

cases=[]
for e in events:
    if e['first_target_candidate_delay'] is None:
        seq=seq_by_id[e['sequence']]
        for minute in (0,20,40):
            frame=min(seq['frames'],key=lambda f:abs(f['timestamp']-seq['official_onset']-minute*60))
            cases.append((frame['file'],f"{seq['id']}\n官方起点 {frame['offset_seconds']/60:+.1f} min",None))
for v in ('calibration_only','learned_residual'):
    for clip in normal[v]:
        if clip['episodes']:
            chosen=max(by_seq[clip['sequence']],key=lambda r:r['scores'][v])
            cases.append((chosen['frame_file'],f"{clip['sequence']}\n{v} peak={chosen['scores'][v]:.3f}",chosen['crop_box']))
fig,axes=plt.subplots(math.ceil(len(cases)/3),3,figsize=(15,3.7*math.ceil(len(cases)/3)),constrained_layout=True)
for ax,(file,title,box) in zip(axes.flat,cases):
    with Image.open(DATA/file) as source:
        source.thumbnail((1100,700))
        ax.imshow(source)
        if box:
            from matplotlib.patches import Rectangle
            w,h=source.size
            ax.add_patch(Rectangle((box[0]*w,box[1]*h),(box[2]-box[0])*w,(box[3]-box[1])*h,fill=False,edgecolor='#e24633',linewidth=2))
    ax.set_title(title,fontsize=9)
    ax.axis('off')
for ax in list(axes.flat)[len(cases):]: ax.axis('off')
fig.savefig(OUT/'diagnostic_cases.png',dpi=140)
plt.close(fig)
print('Written audit.json, comparison.png, diagnostic_cases.png',flush=True)
