"""Recompute experiment-7 metrics without modifying trained outputs or labels."""
from pathlib import Path
from collections import defaultdict, Counter
from statistics import mean, median
from types import SimpleNamespace
import json, math, sys
HERE=Path(__file__).resolve().parent
EXP=HERE.parent
sys.path.insert(0,str(EXP/'scripts'))
import common7 as c
c.configure_cache()
import torch
import numpy as np
ARGS=SimpleNamespace(run_id='exp007_main_20261023',setting='baseline')
BASE=c.folder(ARGS)
SEEDS=(42,43,44)
VARIANTS=('R0','C0','P1','P2','P3','P4','P5')
LABELS=c.labels()
MANIFEST=c.manifest()
SEQS=MANIFEST['sequences']
CFG=c.protocol()

def annotation(r):
    s=LABELS['sequences'][r['sequence']]
    entry=s.get('candidates',{}).get(r['id'])
    if entry is not None and 'label' in entry:
        return entry['label'], bool(entry.get('matches_target')) if entry['label']==1 else False
    frame=s.get('frames',{}).get(Path(r['frame_file']).stem,{})
    if frame.get('status')=='uncertain': return None,False
    if frame.get('status')=='absent': return 0,False
    if r['source']=='hpwren' and s['normal_status']=='confirmed_normal' and frame.get('status')!='visible': return 0,False
    return None,False

def close(a,b):
    assert a is None and b is None or a is not None and b is not None and math.isclose(a,b,abs_tol=1e-8),(a,b)

def fire(rows,v,t):
    result=[]
    for s in SEQS:
        if s['source']!='figlib' or s['split']!='test': continue
        windows=defaultdict(list)
        for r in rows:
            if r['sequence']==s['id'] and r['timestamp']>=s['official_onset']: windows[r['timestamp']].append(r)
        targets=[];crossings=[];alarms=[];wrong=unknown=0
        for stamp,items in sorted(windows.items()):
            for r in items:
                if annotation(r)==(1,True):
                    targets.append(stamp)
                    if r['scores'][v]>=t: crossings.append(stamp)
            winner=max(items,key=lambda r:r['scores'][v])
            if winner['scores'][v]<t: continue
            y=annotation(winner)
            if y==(1,True): alarms.append(stamp)
            elif y[0] is None: unknown+=1
            else: wrong+=1
        minutes=lambda x:(min(x)-s['official_onset'])/60 if x else None
        first=min(alarms) if alarms else None
        human=LABELS['sequences'][s['id']].get('human_visible_timestamp')
        result.append({'sequence':s['id'],'detected':first is not None,'delay_minutes_official':minutes(alarms),
          'delay_minutes_human_visible':(first-human)/60 if first is not None and human is not None else None,
          'first_target_candidate_minutes':minutes(targets),'first_any_target_crossing_minutes':minutes(crossings),
          'wrong_region_positive_windows':wrong,'unknown_positive_winners':unknown})
    return result

def normal(rows,v,t):
    by_time=defaultdict(list)
    for r in rows:
        if r['source']=='hpwren': by_time[(r['sequence'],r['timestamp'])].append(r)
    clips=[];episodes=[]
    for s in SEQS:
        if s['source']!='hpwren' or s['split']!='test': continue
        state=LABELS['sequences'][s['id']]
        if state['normal_status'] in ('excluded','contains_smoke'): continue
        assert state['normal_status']=='confirmed_normal'
        observed=positive=count=0;last=None;frames=[]
        for i,f in enumerate(s['frames']):
            if state['frames'].get(Path(f['file']).stem,{}).get('status') in ('visible','uncertain'): continue
            stamp=f['timestamp']
            end=s['frames'][i+1]['timestamp'] if i+1<len(s['frames']) else s['end_utc_exclusive']
            dt=end-stamp
            if not 0<dt<=CFG['maximum_history_gap_seconds']: continue
            observed+=dt
            items=by_time.get((s['id'],stamp),[])
            r=max(items,key=lambda r:r['scores'][v]) if items else None
            score=r['scores'][v] if r else 0
            if score<t: continue
            positive+=dt
            if last is None or stamp-last>300:
                count+=1
                episodes.append({'sequence':s['id'],'start_timestamp':stamp,'end_timestamp':end,'positive_seconds':0,'positive_frames':0})
            episodes[-1]['end_timestamp']=end
            episodes[-1]['positive_seconds']+=dt
            episodes[-1]['positive_frames']+=1
            frames.append({'id':r['id'],'timestamp':stamp,'duration_seconds':dt,'score':score,
              'original_score':r['scores'].get('R0',r['scores'].get('original')),'age_seconds':r['age_seconds'],
              'length':r['length'],'confidence':r['mean_confidence'],'crop_box':r['crop_box'],'frame_file':r['frame_file']})
            last=stamp
        clips.append({'sequence':s['id'],'observed_seconds':observed,'positive_seconds':positive,'episodes':count,'positive_frames':frames})
    observed=sum(x['observed_seconds'] for x in clips);positive=sum(x['positive_seconds'] for x in clips)
    return {'observed_camera_hours':observed/3600,'false_alarm_episodes':len(episodes),
      'false_alarms_per_camera_hour':len(episodes)/(observed/3600),'positive_minutes':positive/60,
      'positive_time_fraction':positive/observed,'clips':clips,'episodes':episodes}

def distribution(x):
    return {'mean':mean(x),'minimum':min(x),'maximum':max(x),'by_seed':x}

def analyze():
    records=c.load_rows(ARGS)
    test_ids={r['id'] for r in records if r['split']=='test'}
    test_by_id={r['id']:r for r in records if r['split']=='test'}
    by_seed={};details={};training=[];protos=[];events=[];summaries={};checked=0
    for seed in SEEDS:
        d=BASE/f'seed_{seed}'
        summary=c.read_json(d/'summary.json');summaries[seed]=summary
        frozen=c.read_json(d/'thresholds.json');plan=c.read_json(d/'training_plan.json')
        assert plan['smoke_test_only'] is False and set(plan['variants'])==set(VARIANTS)-{'R0'}
        assert summary['evaluation_identity']=={'test_labels_sha256':c.fingerprint('test'),
          'eligibility_sha256':c.digest(c.DATA/'eligibility.json'),'thresholds_sha256':c.digest(d/'thresholds.json')}
        rows=list(c.read_jsonl(d/'test_predictions.jsonl'))
        assert len(rows)==len(test_ids) and {r['id'] for r in rows}==test_ids
        for r in rows:
            assert all(r[k]==val for k,val in test_by_id[r['id']].items())
            assert set(r['scores'])==set(VARIANTS) and all(math.isfinite(x) and 0<=x<=1 for x in r['scores'].values())
        by_seed[seed]={};details[seed]={}
        reference={e['sequence']:e['delay_minutes_official'] for e in summary['variants']['R0']['fire']['per_event']}
        for v in VARIANTS:
            stated=summary['variants'][v];threshold=frozen['variants'][v]['threshold'];close(stated['threshold'],threshold)
            ef=fire(rows,v,threshold);no=normal(rows,v,threshold);details[seed][v]=no
            for k in ('false_alarm_episodes','observed_camera_hours','false_alarms_per_camera_hour','positive_time_fraction','positive_minutes'): close(no[k],stated['normal'][k])
            for actual,expected in zip(ef,stated['fire']['per_event']):
                assert actual['sequence']==expected['sequence']
                for k in actual:
                    if k!='sequence': close(actual[k],expected[k])
                events.append({'seed':seed,'variant':v,**actual})
            delays=[e['delay_minutes_official'] for e in ef if e['detected']]
            close(len(delays),stated['fire']['detected']);close(median(delays),stated['fire']['median_delay_detected_only'])
            close(sum(x<=5 for x in delays)/10,stated['fire']['recall_at_5min'])
            close(sum(x<=10 for x in delays)/10,stated['fire']['recall_at_10min'])
            advances=[reference[e['sequence']]-e['delay_minutes_official'] for e in ef]
            by_seed[seed][v]={'median_delay':median(delays),'mean_delay':mean(delays),'recall5':sum(x<=5 for x in delays)/10,
              'recall10':sum(x<=10 for x in delays)/10,'far':no['false_alarms_per_camera_hour'],
              'positive_fraction':no['positive_time_fraction'],'positive_minutes':no['positive_minutes'],
              'episodes':no['false_alarm_episodes'],'mean_advance':mean(advances),
              'earlier_equal_later':[sum(x>1e-8 for x in advances),sum(abs(x)<=1e-8 for x in advances),sum(x<-1e-8 for x in advances)]}
            if v=='R0': continue
            checkpoint=d/f'{v}.pt';assert c.digest(checkpoint)==frozen['checkpoint_sha256'][v]
            state=torch.load(checkpoint,map_location='cpu',weights_only=True);c.assert_training_identity(state,ARGS);checked+=1
            history=c.read_json(d/f'{v}_history.json');best=min(history,key=lambda h:h['val_event_bce'])
            assert all(math.isfinite(h['train_loss']) and math.isfinite(h['val_event_bce']) for h in history)
            training.append({'seed':seed,'variant':v,'epochs_run':len(history),'best_epoch':best['epoch'],
              'best_validation_bce':best['val_event_bce'],'last_validation_bce':history[-1]['val_event_bce'],
              'train_loss_at_best':best['train_loss'],'last_train_loss':history[-1]['train_loss'],
              'bes_active_batches':sum(h['bes_active_batches'] for h in history),'batches':sum(h['batches'] for h in history)})
            p=c.read_json(d/f'{v}_prototypes.json')
            if 'per_class' in p['train']:
                protos.append({'seed':seed,'variant':v,'train':{k:x['nearest_counts'] for k,x in p['train']['per_class'].items()},
                  'val':{k:x['nearest_counts'] for k,x in p['val']['per_class'].items()},'cosine_gram':p['val']['cosine_gram']})
    metrics=('median_delay','mean_delay','recall5','recall10','far','positive_fraction','positive_minutes','episodes','mean_advance')
    overall={v:{k:distribution([by_seed[s][v][k] for s in SEEDS]) for k in metrics} for v in VARIANTS}
    comparison=[]
    for e in summaries[42]['variants']['R0']['fire']['per_event']:
        seq=e['sequence']
        delays={v:[next(x['delay_minutes_official'] for x in summaries[s]['variants'][v]['fire']['per_event'] if x['sequence']==seq) for s in SEEDS] for v in VARIANTS}
        comparison.append({'sequence':seq,'candidate_floor':e['first_target_candidate_minutes'],'delays_by_seed':delays,
          'mean_delays':{v:mean(x) for v,x in delays.items()}})
    supervision=[]
    for split in ('train','val'):
        accepted=[r for r in records if r['split']==split and r['feature_file'] and annotation(r)[0] in (0,1)
          and (r['source']!='hpwren' or LABELS['sequences'][r['sequence']]['normal_status']=='confirmed_normal')]
        neg=Counter(r['sequence'] for r in accepted if annotation(r)[0]==0)
        supervision.append({'split':split,'candidates':len(accepted),'positive':sum(annotation(r)[0]==1 for r in accepted),
          'negative':sum(annotation(r)[0]==0 for r in accepted),'negative_by_sequence':dict(neg),
          'normal_with_candidates':dict(Counter(r['sequence'] for r in accepted if r['source']=='hpwren')),
          'normal_status':dict(Counter(LABELS['sequences'][s['id']]['normal_status'] for s in SEQS if s['source']=='hpwren' and s['split']==split))})
    legacy=c.read_json(BASE/'legacy_replay/summary.json');oldrows=list(c.read_jsonl(BASE/'legacy_replay/test_predictions.jsonl'));oldout={}
    for v,item in legacy['variants'].items():
        ef=fire(oldrows,v,item['threshold']);no=normal(oldrows,v,item['threshold'])
        for k in ('positive_minutes','false_alarm_episodes','observed_camera_hours'):close(no[k],item['normal'][k])
        for a,b in zip(ef,item['fire']['per_event']):close(a['delay_minutes_official'],b['delay_minutes_official'])
        oldout[v]={'detected':item['fire']['detected'],'median_delay':item['fire']['median_delay_detected_only'],
          'recall5':item['fire']['recall_at_5min'],'far':no['false_alarms_per_camera_hour'],
          'missed':[x['sequence'] for x in ef if not x['detected']]}
    hours=details[42]['R0']['observed_camera_hours']
    return {'created_at':c.now(),'evaluation_tier':CFG['evaluation_tier'],'source_run':str(BASE),
      'seeds':list(SEEDS),'variants':overall,'per_seed':by_seed,'events':events,'event_comparison':comparison,
      'normal_detail':details,'training':training,'prototype_diagnostics':protos,'supervision':supervision,'legacy':oldout,
      'test_camera_hours':hours,'val_camera_hours':c.read_json(BASE/'seed_42/thresholds.json')['variants']['R0']['validation_normal']['observed_camera_hours'],
      'poisson_zero_alarm_upper95':-math.log(.05)/hours,
      'validation':{'verified_checkpoints':checked,'recomputed_event_method_seed_cases':len(events),
       'predictions_per_seed':len(test_ids),'all_record_and_feature_hashes_verified':True,'independent_metric_recomputation':True,
       'test_labels_fingerprint':c.fingerprint('test'),'eligibility_sha256':c.digest(c.DATA/'eligibility.json'),
       'trained_results_unchanged':True}}

def figures(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=font_manager.FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'font.size':11,'axes.unicode_minus':False,
      'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,8.4),layout='constrained');x=np.arange(7)
    panels=[('recall5',100,'5分钟内检出率（越高越好）','%'),('mean_delay',1,'10个事件平均延迟（越低越好）','分钟'),
      ('far',1,'误报频率（越低越好）','误报段 / 摄像头小时'),('positive_fraction',100,'正常视频超阈时间占比（越低越好）','%')]
    for ax,(key,mult,title,unit) in zip(axes.flat,panels):
        ys=np.array([a['variants'][v][key]['by_seed'] for v in VARIANTS])*mult
        ax.bar(x,ys.mean(1),color=['#415c77']+['#a8bfd4']*6,width=.65)
        for j,m in enumerate(('o','^','s')):
            ax.scatter(x[1:]+(j-1)*.13,ys[1:,j],marker=m,color='#27394b',s=26,zorder=3,label=f'种子 {SEEDS[j]}')
        ax.scatter([0],[ys[0,0]],color='#27394b',s=26,zorder=3)
        ax.set(xticks=x,xticklabels=VARIANTS,title=title,ylabel=unit,ylim=(0,None))
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        if key=='recall5':ax.set_ylim(0,100);ax.legend(frameon=False,ncol=3,fontsize=9,loc='upper right')
        if key=='far':ax.axhline(.1,color='#a24b39',ls=':',lw=1)
        if key=='positive_fraction':ax.axhline(1,color='#a24b39',ls=':',lw=1)
    fig.suptitle('实验七：PSL区域适配的收益与代价',fontsize=16)
    fig.supxlabel('柱为三个种子的平均，点为各次结果。10个火情事件、同一批9.974小时正常视频；不是30个独立事件。',fontsize=10)
    fig.savefig(HERE/'method_comparison.png',dpi=160);plt.close(fig)
    mat=[];names=[]
    for item in a['event_comparison']:
        mat.append([item['candidate_floor'],item['delays_by_seed']['R0'][0],*item['delays_by_seed']['P5']])
        name=item['sequence']
        names.append('2019-10-05 弱烟' if name.startswith('20191005') else '2020-03-06 弱烟' if name.startswith('20200306') else name.split('_')[1])
    fig,ax=plt.subplots(figsize=(9.8,6.8),layout='constrained')
    im=ax.imshow(mat,cmap='YlOrRd',vmin=0,vmax=max(max(r) for r in mat),aspect='auto')
    for i,row in enumerate(mat):
        for j,value in enumerate(row):ax.text(j,i,f'{value:.2f}',ha='center',va='center',color='white' if value>15 else '#272727',fontsize=11)
    ax.set_xticks(range(5),['首个目标候选','原版 R0','P5 / 42','P5 / 43','P5 / 44'])
    ax.set_yticks(range(10),names);ax.set_title('逐场景延迟：弱烟提前与个别场景退化同时存在',pad=15)
    fig.colorbar(im,ax=ax,label='相对官方起点的分钟数')
    fig.savefig(HERE/'per_event_delays.png',dpi=160);plt.close(fig)

def main():
    a=analyze();c.write_json(HERE/'analysis.json',a)
    import csv
    with (HERE/'method_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['variant','mean_of_seed_medians_min','mean_event_delay_min','mean_recall5','mean_recall10','mean_far','mean_positive_fraction','alarm_counts'])
        for v,d in a['variants'].items():writer.writerow([v,*[d[k]['mean'] for k in ('median_delay','mean_delay','recall5','recall10','far','positive_fraction')],json.dumps(d['episodes']['by_seed'])])
    with (HERE/'event_details.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(a['events'][0]));writer.writeheader();writer.writerows(a['events'])
    figures(a)
    print(json.dumps({'validation':a['validation'],'method_table':{v:{k:round(d[k]['mean'],4) for k in ('median_delay','mean_delay','recall5','far','positive_fraction')} for v,d in a['variants'].items()},
      'supervision':a['supervision']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()

