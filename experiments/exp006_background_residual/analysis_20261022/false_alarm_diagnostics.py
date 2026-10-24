"""Frozen-head false-alarm diagnostics; no training or threshold selection.

The attribution baseline is the mean input vector of the 257 labelled training
background candidates. Exact integrated gradients are computed analytically for
the Linear-ReLU-Linear fusion head. Feature replacement is a model sensitivity
diagnostic, not an estimate of a physical cause.
Run with --save to write false_alarm_diagnostics.json beside this script.
"""
from __future__ import annotations
import collections
import json
import math
import os
from pathlib import Path
import sys
import zipfile

OUT=Path(__file__).resolve().parent
EXP=OUT.parent
ROOT=EXP.parents[1]
RUN=EXP/'outputs/exp006_main_20261021/baseline'
SEED=RUN/'seed_42'
DATA=ROOT/'data/exp006_background_residual'
os.environ['MPLCONFIGDIR']=str(OUT/'render_cache')
os.environ['TEMP']=os.environ['TMP']=str(ROOT/'.cache/exp006/temp')
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT.parent/'pyronear_lab/.venv/Lib/site-packages'))
sys.path.insert(0,str(EXP/'scripts'))
import numpy as np
import torch
from torch.nn import functional as F
from learning import BackgroundPredictor,Fusion,base_features,candidate_label

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def sigmoid(z): return 1/(1+np.exp(-np.asarray(z)))

meta=read(RUN/'run.json')
labels=read(DATA/'labels.json')
thresholds=read(SEED/'thresholds.json')
manifest=read(DATA/'manifest.json')
seqs={s['id']:s for s in manifest['sequences']}
test=lines(SEED/'test_predictions.jsonl')
raw=[r for sid in meta['sequence_ids'] for r in lines(RUN/f'{sid}.jsonl')]
train_background=[r for r in raw if r['split']=='train' and r['feature_file'] and candidate_label(r,labels)==0]
normal=[r for r in test if r['source']=='hpwren' and r['feature_file']]
assert len(train_background)==257
device='cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(4)

def load_x(rows):
    return torch.from_numpy(np.stack([np.load(RUN/r['feature_file'])['features'].astype(np.float32) for r in rows])).to(device)

def assemble(model,rows):
    inputs=[]
    norms=[]
    with torch.inference_mode():
        for start in range(0,len(rows),64):
            batch=rows[start:start+64]
            x=F.normalize(load_x(batch),dim=-1)
            pieces=[]
            mag=[]
            k=model.cfg['history_frames']
            for j in range(k,k+model.cfg['residual_steps']):
                h=x[:,j-k:j]
                ref=F.normalize(h.mean(1),dim=-1) if model.variant=='mean_residual' else model.predictor(h)
                residual=x[:,j]-ref
                n=residual.norm(dim=-1)
                pieces.extend([residual.mean(1),n.mean(1,keepdim=True),n.amax(1,keepdim=True)])
                mag.append(torch.stack([n.mean(1),n.amax(1)],dim=-1))
            base=torch.from_numpy(np.stack([base_features(r) for r in batch])).to(device)
            inputs.append(torch.cat([*pieces,base],-1).cpu().numpy())
            norms.append(torch.stack(mag,dim=1).cpu().numpy())
    return np.concatenate(inputs),np.concatenate(norms)

def weights(model):
    s=model.head.state_dict()
    return [s[k].cpu().numpy().astype(np.float64) for k in ('0.weight','0.bias','2.weight','2.bias')]

def zscore(x,w):
    a,b,v,c=w
    return (np.maximum(x@a.T+b,0)@v.T+c).reshape(-1)

def analytic_ig(x,reference,w):
    a,b,v,c=w
    a0=reference@a.T+b
    a1=x@a.T+b
    fraction=np.zeros_like(a1)
    fraction[(a0>0)&(a1>0)]=1
    rising=(a0<=0)&(a1>0)
    falling=(a0>0)&(a1<=0)
    frac_rise=np.divide(a1,a1-a0,out=np.zeros_like(a1),where=np.abs(a1-a0)>1e-15)
    frac_fall=np.divide(a0,a0-a1,out=np.zeros_like(a1),where=np.abs(a1-a0)>1e-15)
    fraction[rising]=frac_rise[rising]
    fraction[falling]=frac_fall[falling]
    effective=(fraction*v.reshape(1,-1))@a
    result=(x-reference)*effective
    error=np.max(np.abs(result.sum(1)-(zscore(x,w)-zscore(reference[None],w)[0])))
    assert error<1e-8,error
    return result,error

groups={
 'residual_direction':np.concatenate([np.arange(j*386,j*386+384) for j in range(3)]),
 'residual_mean_magnitude':np.array([384,770,1156]),
 'residual_max_magnitude':np.array([385,771,1157]),
 'classifier_logit':np.array([1158]),
 'track_length':np.array([1159]),
 'detector_confidence':np.array([1160]),
 'candidate_count':np.array([1161]),
}
groups['all_residual']=np.arange(1158)

result={
 'scope':'Frozen baseline/seed_42; independently reviewed normal clips only; no training, threshold or label changes.',
 'interpretation':'Signed logit attributions use a training-background-mean input reference. Positive values raise the output relative to that reference. Replacing feature blocks can create inputs outside the observed distribution and is sensitivity analysis, not physical causality.',
 'reference_background_candidates':len(train_background),'normal_feature_candidates':len(normal),'variants':{},
 'explicit_base_attributes':['classifier logit / 10','log(1 + track length)','mean detector confidence','candidate count / 10'],
 'missing_explicit_attributes':['age since last real detection','current frame supported by a real detection','optical flow','physical smoke opacity'],
}

for variant in ('mean_residual','learned_residual'):
    st=torch.load(SEED/f'{variant}.pt',map_location='cpu',weights_only=True)
    predictor=BackgroundPredictor(st['dimension'],st['config']['hidden_dim']) if variant=='learned_residual' else None
    model=Fusion(variant,st['dimension'],st['config'],predictor).to(device).eval()
    model.load_state_dict(st['state'])
    tr_x,_=assemble(model,train_background)
    te_x,mags=assemble(model,normal)
    reference=tr_x.astype(np.float64).mean(0)
    w=weights(model)
    z=zscore(te_x,w)
    p=sigmoid(z)
    maxdiff=max(abs(float(p[i])-r['scores'][variant]) for i,r in enumerate(normal))
    assert maxdiff<2e-6,maxdiff
    attribution,error=analytic_ig(te_x.astype(np.float64),reference,w)
    replacement={}
    for name,idx in groups.items():
        x=te_x.astype(np.float64).copy()
        x[:,idx]=reference[idx]
        replacement[name]=sigmoid(zscore(x,w))
    vr=dict(reference_probability=float(sigmoid(zscore(reference[None],w))[0]),score_max_abs_diff=maxdiff,
            ig_completeness_max_error=error,threshold=thresholds['variants'][variant]['threshold'],normal_rows=[],by_clip={})
    for i,r in enumerate(normal):
        width,height=seqs[r['sequence']]['frames'][0]['width'],seqs[r['sequence']]['frames'][0]['height']
        b=r['crop_box']
        vr['normal_rows'].append(dict(id=r['id'],sequence=r['sequence'],timestamp=r['timestamp'],
            probability=float(p[i]),score_logit=float(z[i]),alarm=float(p[i])>=vr['threshold'],
            logit=r['logit'],length=r['length'],mean_confidence=r['mean_confidence'],n_tubes=r['n_tubes'],
            age_seconds=r['age_seconds'],crop_pixels=[round((b[2]-b[0])*width),round((b[3]-b[1])*height)],
            residual_magnitude_by_time=mags[i].tolist(),
            logit_attribution={name:float(attribution[i,idx].sum()) for name,idx in groups.items()},
            probability_after_reference_replacement={name:float(v[i]) for name,v in replacement.items()}))
    for sid in sorted({r['sequence'] for r in normal}):
        ss=[r for r in vr['normal_rows'] if r['sequence']==sid]
        fp=[r for r in ss if r['alarm']]
        vr['by_clip'][sid]=dict(rows=len(ss),positive_rows=len(fp),
            average_magnitude_all=float(np.mean([np.mean(np.array(r['residual_magnitude_by_time'])[:,0]) for r in ss])),
            average_magnitude_positive=float(np.mean([np.mean(np.array(r['residual_magnitude_by_time'])[:,0]) for r in fp])) if fp else None,
            mean_fp_attribution={name:float(np.mean([r['logit_attribution'][name] for r in fp])) for name in groups} if fp else {},
            remaining_positive_rows_after_reference_replacement={name:sum(r['probability_after_reference_replacement'][name]>=vr['threshold'] for r in fp) for name in groups})
    result['variants'][variant]=vr
    print('VARIANT',variant,'reference_p',vr['reference_probability'],'score_error',maxdiff,flush=True)
    for sid,v in vr['by_clip'].items():print('CLIP',sid,json.dumps(v),flush=True)
    for r in vr['normal_rows']:
        if r['alarm'] and 'buff' in r['sequence']:
            print('BUFF_CASE',json.dumps(r),flush=True)

cal=torch.load(SEED/'calibration_only.pt',map_location='cpu',weights_only=True)
cw=cal['state']['head.weight'].numpy().reshape(-1).astype(float)
cb=float(cal['state']['head.bias'].item())
with zipfile.ZipFile(ROOT.parent/'pyronear_lab/models/v0.6.0/model.zip') as zz:
    original_cal=json.loads(zz.read('logistic_calibrator.json'))
result['linear_calibration']=dict(new_weights_on_scaled_inputs=cw.tolist(),new_bias=cb,original=original_cal,examples=[])
for sid in ('normal_20261016_Q4_sm-n-mobo-c','normal_20261008_Q4_buff-s-mobo-c'):
    candidates=[r for r in normal if r['sequence']==sid]
    r=max(candidates,key=lambda r:r['scores']['learned_residual'])
    fv=base_features(r).astype(float)
    original_features=np.array([r['logit'],math.log1p(r['length']),r['mean_confidence'],r['n_tubes']])
    old_contrib=original_features*np.array(original_cal['coefficients'])
    assert abs(float(sigmoid(old_contrib.sum()+original_cal['intercept']))-r['official_probability'])<1e-7
    entry=dict(id=r['id'],new_logit_contributions=dict(zip(('classifier_logit','track_length','detector_confidence','candidate_count'),(fv*cw).tolist())),
        new_bias=cb,new_probability=float(sigmoid((fv*cw).sum()+cb)),original_probability=r['official_probability'],
        original_logit_contributions=dict(zip(('classifier_logit','track_length','detector_confidence','candidate_count'),old_contrib.tolist())),original_bias=original_cal['intercept'])
    result['linear_calibration']['examples'].append(entry)
    print('LINEAR_EXAMPLE',json.dumps(entry),flush=True)

if '--save' in sys.argv:
    target=OUT/'false_alarm_diagnostics.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('SAVED',target)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib import font_manager
    from PIL import Image
    font=Path(r'C:\Windows\Fonts\msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(14,8.4),constrained_layout=True)
    grid=fig.add_gridspec(2,2,height_ratios=[1.65,1])
    display=[('residual_direction','特征变化组合'),('residual_mean_magnitude','平均变化幅度'),('residual_max_magnitude','最大局部变化幅度'),
             ('classifier_logit','原时序分数'),('track_length','轨迹长度'),('detector_confidence','检测置信度'),('candidate_count','候选数量')]
    learned=result['variants']['learned_residual']
    for column,(sid,title) in enumerate((('normal_20261016_Q4_sm-n-mobo-c','sm：19个连续超阈时刻（1段误报）'),('normal_20261008_Q4_buff-s-mobo-c','buff：4个超阈时刻（4段误报）'))):
        ax=fig.add_subplot(grid[0,column])
        attr=learned['by_clip'][sid]['mean_fp_attribution']
        vals=[attr[name] for name,_ in display]
        ax.barh(range(len(display)),vals,color=['#cc704b' if v>=0 else '#438b9a' for v in vals])
        ax.set(yticks=range(len(display)),yticklabels=[label for _,label in display],xlim=(-.2,1.35),title=title,
               xlabel='相对训练背景参照的平均输出贡献（logit）')
        ax.invert_yaxis()
        ax.axvline(0,color='#56606a',linewidth=.8)
        for i,v in enumerate(vals):ax.text(v+(.02 if v>=0 else -.02),i,f'{v:+.3f}',va='center',ha='left' if v>=0 else 'right',fontsize=9)
    crops=grid[1,:].subgridspec(1,5)
    original_by_id={r['id']:r for r in test}
    chosen=[max([r for r in learned['normal_rows'] if 'Q4_sm-' in r['sequence']],key=lambda r:r['probability'])]
    chosen += [r for r in learned['normal_rows'] if 'buff' in r['sequence'] and r['alarm']]
    for i,r in enumerate(chosen):
        ax=fig.add_subplot(crops[0,i])
        source_row=original_by_id[r['id']]
        with Image.open(DATA/source_row['frame_file']) as im:
            w,h=im.size
            box=source_row['crop_box']
            region=im.crop(tuple(round(v*s) for v,s in zip(box,(w,h,w,h))))
            ax.imshow(region)
        label='sm峰值' if i==0 else f'buff #{i}'
        ax.set_title(f"{label}；分数 {r['probability']:.3f}\n长度 {r['length']}；证据年龄 {r['age_seconds']}秒\n裁剪 {r['crop_pixels'][0]}×{r['crop_pixels'][1]}像素",fontsize=9)
        ax.axis('off')
    fig.suptitle('学习残差误报：属性影响与真实候选裁剪\n正值推高分数，负值压低分数；归因依赖参照，不能直接解释为颜色、云或曝光的物理因果',fontsize=14)
    fig.savefig(OUT/'false_alarm_attributes.png',dpi=160)
    plt.close(fig)
    print('SAVED',OUT/'false_alarm_attributes.png')
