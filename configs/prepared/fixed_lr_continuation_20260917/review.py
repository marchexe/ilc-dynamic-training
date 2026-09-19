"""Read-only continuation status/verification; analysis writes only derived reports."""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from validation.verify_fixed_lr import verify
from training.runtime import sha256
PREP=Path(__file__).resolve().parent
PLAN=json.loads((PREP/'preflight.json').read_text())
OLD=ROOT/'runs/pbt/foundation_fixed_lr_50epochs_20260916'
BASE=json.loads((OLD/'manifest.json').read_text())
METRIC=BASE['config']['pbt']['metric']
WORKING=sorted(k for k in BASE['initial_evaluation']['metrics'] if '_mistag_eff_' in k)
NAMES=list(PLAN['arms'])


def manifests():
    return {n:json.loads((Path(a['run'])/'manifest.json').read_text()) for n,a in PLAN['arms'].items()}


def status():
    rows=[]
    for n,a in PLAN['arms'].items():
        path=Path(a['run'])/'manifest.json'
        m=json.loads(path.read_text()) if path.exists() else {}
        done=[g for g in m.get('generations',[]) if g.get('status')=='completed']
        rows.append(dict(arm=n,gpu=a['gpu'],status=m.get('status','not_started'),
                         completed_additional_epochs=len(done),full_epoch=50+len(done),
                         checkpoint_epoch=done[-1]['epoch'] if done else 67,failure=m.get('failure')))
    evaluation=ROOT/'runs/eval/original_pretrained_0_19_20260917'
    records=[json.loads(p.read_text()) for p in evaluation.glob('epoch-*/result.json')]
    print(json.dumps(dict(continuations=rows,original_evaluations={s:sum(r.get('status')==s for r in records)
                     for s in ['completed','running','failed']}),indent=2))


def verify_all():
    ms=manifests()
    incomplete=[]
    for n,m in ms.items():
        completed=[g for g in m.get('generations',[]) if g.get('status')=='completed'
                   and g.get('workers',{}).get(n,{}).get('status')=='completed'
                   and g['workers'][n].get('metrics')]
        if m.get('status')!='completed' or len(completed)!=50:
            incomplete.append(dict(arm=n,status=m.get('status'),completed_full_epoch=50+len(completed),
                                   last_checkpoint=completed[-1]['epoch'] if completed else 67,
                                   next_generation=m.get('next_generation'),failure=m.get('failure')))
    if incomplete:
        result=dict(passed=False,incomplete_runs=incomplete,
                    message='Full comparison requires both arms through full epoch 100. Completed checkpoints are retained.')
        print(json.dumps(result,indent=2))
        return result
    reports={n:verify(Path(a['run'])) for n,a in PLAN['arms'].items()}
    failures=[f'{n}: {f}' for n,r in reports.items() for f in r['failures']]
    def check(label,ok):
        if not ok:failures.append(label)
    for n,m in ms.items():
        a=PLAN['arms'][n]
        check(n+': unchanged contract',m['fingerprint']==a['fingerprint'])
        for component,digest in a['hashes'].items():
            check(n+': own original '+component,sha256(Path(a['run'])/n/f'net_epoch-67_{component}.pt')==digest)
        initial=m['initial_evaluation']['metrics']
        previous=BASE['generations'][-1]['workers'][n]['metrics']
        for key in ['validation_loss',METRIC,*WORKING]:check(n+': epoch-50 boundary '+key,initial[key]==previous[key])
        check(n+': boundary predictions',initial['validation_data_audit']['prediction_sha256']==previous['validation_data_audit']['prediction_sha256'])
    for index in range(50):
        audits=[]
        for n,m in ms.items():
            g=next(g for g in m['generations'] if g['index']==index)
            check(n+f'/{index}: uninterrupted numbering/seed',g['epoch']==68+index and g['seed']==12395+index)
            train=g['workers'][n]['metrics']['train_data_audit']
            audits.append([train['dataset'],train['consumed'],train['traversal']])
        check(f'epoch {51+index}: cross-arm training sequence',audits[0]==audits[1])
    snapshot={str(p.relative_to(OLD)):[p.stat().st_size,p.stat().st_mtime_ns] for p in OLD.rglob('*') if p.is_file()}
    check('completed 50-epoch run untouched',snapshot==json.loads((PREP/'source_artifacts_snapshot.json').read_text()))
    result=dict(passed=not failures,failures=failures,arms=reports)
    print(json.dumps(result,indent=2))
    if failures:raise RuntimeError('Morning verification failed')
    return result


def write_analysis(records,out):
    """Records contain real epoch metrics and source checkpoint paths."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(parents=True,exist_ok=True)
    def csv_file(name,rows):
        with (out/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    csv_file('epochs.csv',records)
    summaries=[];rolling=[];crossovers={};stable={}
    for key in [METRIC,*WORKING]:
        series={n:sorted([r for r in records if r['arm']==n],key=lambda r:r['full_epoch']) for n in NAMES}
        for n,rows in series.items():
            for window in [5,10,20]:
                for i in range(window-1,len(rows)):
                    rolling.append(dict(arm=n,metric=key,full_epoch=rows[i]['full_epoch'],window=window,
                                        mean=float(np.mean([r[key] for r in rows[i-window+1:i+1]]))))
            for start in [1,51]:
                subset=[r for r in rows if r['full_epoch']>=start]
                if not subset:continue
                best=min(subset,key=lambda r:r[key]);y=np.array([r[key] for r in subset])
                row=dict(arm=n,metric=key,range_start=start,range_end=subset[-1]['full_epoch'],
                         best=float(y.min()),best_full_epoch=best['full_epoch'],best_checkpoint=best['checkpoint'],final=float(y[-1]))
                for window in [5,10,20]:
                    if len(subset)<window:continue
                    late=subset[-window:];v=np.array([r[key] for r in late]);x=np.array([r['full_epoch'] for r in late])
                    row.update({f'mean_last_{window}':float(v.mean()),f'slope_last_{window}_pp_per_epoch':float(np.polyfit(x,v,1)[0]),
                                f'std_last_{window}':float(v.std(ddof=1))})
                if len(y)>=20:
                    row['last10_minus_previous10_pp']=float(y[-10:].mean()-y[-20:-10].mean())
                    changes=[row['slope_last_10_pp_per_epoch'],row['slope_last_20_pp_per_epoch'],row['last10_minus_previous10_pp']]
                    row['late_trend']='improving' if all(v<0 for v in changes) else 'degraded' if all(v>0 for v in changes) else 'mixed_or_flat'
                summaries.append(row)
        epochs=sorted(set(r['full_epoch'] for r in series[NAMES[0]]) & set(r['full_epoch'] for r in series[NAMES[1]]))
        lookup={n:{r['full_epoch']:r[key] for r in series[n]} for n in NAMES}
        signs=[int(np.sign(lookup[NAMES[0]][e]-lookup[NAMES[1]][e])) for e in epochs]
        crossovers[key]=[dict(full_epoch=epochs[i],winner=NAMES[0] if signs[i]<0 else NAMES[1] if signs[i]>0 else 'tie')
                         for i in range(1,len(epochs)) if signs[i]!=signs[i-1]]
        means={n:float(np.mean([lookup[n][e] for e in epochs[-20:]])) for n in NAMES}
        stable[key]=dict(window=20,means=means,best_observed_mean=min(means,key=means.get),
                         final20_wins={n:sum(lookup[n][e]<lookup[NAMES[1-NAMES.index(n)]][e] for e in epochs[-20:]) for n in NAMES})
    # Stable field sets for CSV even when testing the writer on a short real prefix.
    fields=sorted({k for r in summaries for k in r})
    csv_file('summary.csv',[{k:r.get(k) for k in fields} for r in summaries]);csv_file('rolling_means.csv',rolling)
    def plot(keys,filename,from_epoch):
        fig,axes=plt.subplots(len(keys),1,figsize=(11,3*len(keys)),squeeze=False)
        for ax,key in zip(axes[:,0],keys):
            for n in NAMES:
                rows=[r for r in records if r['arm']==n and r['full_epoch']>=from_epoch]
                ax.plot([r['full_epoch'] for r in rows],[r[key] for r in rows],label=n)
            ax.axvline(50.5,color='gray',linestyle=':');ax.set_ylabel(key.replace('validation_',''));ax.grid(alpha=.2);ax.legend()
        axes[-1,0].set_xlabel('Full epoch since common pretrained epoch 17')
        fig.tight_layout();fig.savefig(out/filename,dpi=160);plt.close(fig)
    plot([METRIC],'mistag_epochs_1_100.png',1);plot([METRIC],'mistag_epochs_51_100.png',51)
    plot(WORKING,'working_points.png',1)
    threshold=BASE['generations'][-1]['workers']['lr_14e-6']['metrics'][METRIC]
    convergence={n:next((r['full_epoch'] for r in records if r['arm']==n and r[METRIC]<=threshold),None) for n in NAMES}
    best=min(records,key=lambda r:r[METRIC])
    report=dict(best_single_checkpoint=best,late_trajectory=stable,crossovers=crossovers,
                first_epoch_reaching_original_14e6_final=dict(threshold_percent=threshold,epochs=convergence),
                interpretation='Use rolling 5/10/20 means, slopes, and variance jointly; raw best checkpoints and '
                'stable late means answer different questions. Late degradation/plateau is descriptive on this '
                'fixed validation sample, not an independent significance claim. No PBT conclusion is implied.')
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    print('Analysis:',out)
    return report


def analyze():
    if not verify_all()['passed']:
        raise RuntimeError('Analysis deferred: continuation is incomplete; see run status above')
    records=[]
    for n,m in manifests().items():
        for source,run in [(BASE,OLD),(m,Path(PLAN['arms'][n]['run']))]:
            for g in source['generations']:
                metrics=g['workers'][n]['metrics']
                records.append(dict(arm=n,full_epoch=g['epoch']-17,checkpoint_epoch=g['epoch'],
                                    checkpoint=str(run/n/f'net_epoch-{g["epoch"]}_state.pt'),
                                    **{k:metrics[k] for k in ['validation_loss',METRIC,*WORKING]}))
    return write_analysis(records,PREP/'analysis')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['status','verify','analyze'])
    args=parser.parse_args()
    try:
        result={'status':status,'verify':verify_all,'analyze':analyze}[args.action]()
        if args.action=='verify' and not result['passed']:
            raise SystemExit(1)
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        raise SystemExit(f'error: {error}')
