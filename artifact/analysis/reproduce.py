"""Offline analysis from package data; never accesses the original project or APIs."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[name]='1'
import argparse, hashlib, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent/'rq3'))
from workspace import output_directory,input_digest
OUT=output_directory()
OUT.mkdir(parents=True,exist_ok=True)

def read(path):
    source=ROOT/path
    if source.suffix=='.xlsx':return pd.read_excel(source,engine='openpyxl')
    return pd.read_csv(source,encoding='utf-8-sig',float_precision='round_trip')
def save(name,rows):
    d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    d.to_csv(OUT/(name+'.csv'),index=False)
    return d
def yes(s):return s.astype(str).str.lower().isin(['1','true','1.0'])
def jsonsave(path,obj):
    (OUT/Path(path).name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def executions():
    r=read('data/execution_results.xlsx')
    r['model']=r.model.str.lower();r['implementation']=r['mode'].str.upper()
    r['execution_id']=r.model+'/'+r.implementation+'/'+r.task_id
    r['success']=r.success_corrected.astype(int)
    assert len(r)==1392 and r.execution_id.nunique()==1392 and r.success.sum()==770
    return r

def rq2_labels():
    q=read('data/rq2_annotations.xlsx')
    keys=q.execution_id.str.split('/',n=2,expand=True)
    q['model']=keys[0];q['mode']=keys[1].str.lower();q['task_id']=keys[2]
    return q

def rq3_frame():
    """Derive analysis fields from final labels, execution outcomes and four GUI features."""
    keys=['trajectory_id','source_step_index']
    f=read('data/rq3_annotations.xlsx').merge(read('data/gui_features/rq3_steps.csv'),on=keys,validate='one_to_one')
    assert len(f)==17658
    r=executions()[['execution_id','model','implementation','task_id','success','optimal_steps']]
    f=f.merge(r,left_on='trajectory_id',right_on='execution_id',validate='many_to_one')
    f=f.sort_values(keys).reset_index(drop=True)
    f['backbone']=f.model;f['configuration']=f.implementation;f['final_success']=f.success
    f['system']=f.model+'/'+f.implementation
    histories=[]
    for tid,g in f.groupby('trajectory_id',sort=True):
        nE=nN=streak=0;previous=None;last=None
        for row in g.itertuples():
            step=int(row.source_step_index);adjacent=last is not None and step==last+1
            histories.append(dict(trajectory_id=tid,source_step_index=step,
                prior_error_fraction=nE/(nE+nN) if nE+nN else np.nan,
                previous_error=int(previous=='E') if adjacent and previous in ['E','N'] else np.nan,
                excess_streak=max(streak-1,0) if adjacent else 0))
            if not adjacent:streak=0
            if row.label=='E':nE+=1;streak+=1
            elif row.label=='N':nN+=1;streak=0
            else:streak=0
            previous=row.label;last=step
    f=f.merge(pd.DataFrame(histories),on=keys,validate='one_to_one')
    f['y']=f.label.map({'E':1,'N':0})
    f['eligible_primary']=f.y.notna()&f.previous_error.notna()&f.prior_error_fraction.notna()
    f['eligible_onset']=f.eligible_primary&f.previous_error.eq(0)
    f['eligible_persistence']=f.eligible_primary&f.previous_error.eq(1)
    d=f[f.eligible_primary]
    for raw,z in [('source_step_index','position_z'),('prior_error_fraction','fraction_z'),('excess_streak','excess_z')]:
        f[z]=(f[raw]-d[raw].mean())/d[raw].std(ddof=0)
    for feature in ['visual_stagnation','median_candidate_area','spatial_crowding','edge_density_clutter']:
        scaling=f[f.y.notna()].groupby('system')[feature].agg(['mean',lambda s:s.std(ddof=0)])
        scaling.columns=['mean','sd']
        f[feature+'_z']=(f[feature]-f.system.map(scaling['mean']))/f.system.map(scaling.sd)
    return f

def basic():
    r=executions()
    grouped=r.groupby(['model','implementation','task_difficulty']).success.agg(['sum','count']).reset_index()
    grouped['rate']=grouped['sum']/grouped['count'];save('rq1_completion',grouped)
    records=[]
    from itertools import combinations
    for model,g in r.groupby('model'):
        p=g.pivot(index='task_id',columns='implementation',values='success')
        for a,b in combinations(p.columns,2):
            records.append(dict(model=model,first=a,second=b,SS=int(((p[a]==1)&(p[b]==1)).sum()),SF=int(((p[a]==1)&(p[b]==0)).sum()),FS=int(((p[a]==0)&(p[b]==1)).sum()),FF=int(((p[a]==0)&(p[b]==0)).sum())))
    save('rq1_pairs',records)
    coverage=r.groupby('task_id').success.sum();save('rq1_task_coverage',coverage.rename('successful_systems').reset_index())
    q=rq2_labels();assert len(q)==572
    q['code']=q.dominant_code;q['family']=q.code.str[0]
    counts=q.groupby('code').size().rename('count').reset_index();counts['fraction']=counts['count']/len(q);save('rq2_causes',counts)
    profiles=q.groupby(['model','mode','code']).size().rename('count').reset_index()
    profiles['diagnostic_n']=profiles.groupby(['model','mode'])['count'].transform('sum')
    profiles['percentage']=100*profiles['count']/profiles.diagnostic_n;save('rq2_system_profiles',profiles)
    pairs=[]
    for row in q.to_dict('records'):
        second=row.get('cooccurring_code')
        if pd.notna(second) and str(second).strip():pairs.append('+'.join(sorted([row['code'],str(second)])))
    save('rq2_cooccurrence',pd.Series(pairs).value_counts().rename_axis('pair').reset_index(name='count'))
    steps=rq3_frame();labels=read('data/rq3_annotations.xlsx')
    assert labels.label.value_counts().to_dict()=={'N':13380,'E':4194,'U':84}
    joined=steps.rename(columns={'label':'label_frozen'})
    assert len(joined)==len(labels)
    outcomes=r.set_index('execution_id').success.to_dict();rows=[];history=[]
    for tid,g in joined.groupby('trajectory_id',sort=True):
        g=g.sort_values('source_step_index');seq=g.label_frozen.tolist();positions=g.source_step_index.astype(int).tolist()
        nE=nN=streak=maximum=runs=ee=en=ne=nn=0;previous=None;lastpos=None;seen_error=False;nonerror_since=False;recurrence=False
        for k,label in zip(positions,seq):
            adjacent=lastpos is not None and k==lastpos+1
            previous_error=(1 if previous=='E' else 0) if adjacent and previous in ['E','N'] else np.nan
            history.append(dict(trajectory_id=tid,source_step_index=k,prior_error_fraction=nE/(nE+nN) if nE+nN else np.nan,previous_error=previous_error,excess_streak=max(streak-1,0) if adjacent else 0))
            if not adjacent:streak=0;seen_error=False;nonerror_since=False
            if label=='E':
                if seen_error and nonerror_since:recurrence=True
                seen_error=True;nonerror_since=False
                if not (adjacent and previous=='E'):runs+=1
                streak+=1;nE+=1;maximum=max(maximum,streak)
            elif label=='N':
                nN+=1;streak=0
                if seen_error:nonerror_since=True
            else:streak=0;seen_error=False;nonerror_since=False
            if adjacent:
                ee+=previous=='E' and label=='E';en+=previous=='E' and label=='N';ne+=previous=='N' and label=='E';nn+=previous=='N' and label=='N'
            previous=label;lastpos=k
        rows.append(dict(trajectory_id=tid,success=outcomes[tid],E=nE,N=nN,U=seq.count('U'),error_fraction=nE/(nE+nN) if nE+nN else np.nan,max_error_streak=maximum,error_runs=runs,EE=ee,EN=en,NE=ne,NN=nn,recurrence=recurrence))
    trajectory=save('rq3_trajectories',rows);h=pd.DataFrame(history)
    check=steps.merge(h,on=['trajectory_id','source_step_index'],suffixes=('','_new'),validate='one_to_one')
    for c in ['prior_error_fraction','previous_error','excess_streak']:
        assert np.allclose(check[c],check[c+'_new'],equal_nan=True),c
    summary=[]
    for outcome,g in trajectory.groupby('success'):
        summary.append(dict(success=int(outcome),n=len(g),any_error_n=int(g.E.gt(0).sum()),any_error_fraction=g.E.gt(0).mean(),mean_error_fraction=g.error_fraction.mean(),mean_max_streak=g.max_error_streak.mean(),EE_n=int(g.EE.gt(0).sum()),EN_n=int(g.EN.gt(0).sum()),recurrence_n=int(g.recurrence.sum()),mean_error_runs=g.error_runs.mean()))
    save('rq3_summary',summary)
    # Whole-cohort terminal reference windows, with fixed common task support per curve.
    curves=[];windows=[]
    for D,W in [(d,w) for d in [5,10,15] for w in [5,10,15]]:
        support=r[r.optimal_steps+D+W<=30]
        if support.empty:continue
        for d in range(D+1):
            active=support[support.step_count>support.optimal_steps+d].copy();N=len(active);last=0;gains=[]
            for w in range(1,W+1):
                n=int((active.success.eq(1)&active.step_count.le(active.optimal_steps+d+w)).sum())
                inc=n-last;gains.append(inc/N if N else np.nan);last=n
                windows.append(dict(scope=f'D{D}_W{W}',d=d,w=w,tasks=support.task_id.nunique(),active_n=N,completion_n=n,completion_rate=n/N if N else np.nan,new_completion_n=inc,gain=inc/N if N else np.nan))
            rho=float(stats.spearmanr(range(1,W+1),gains).statistic) if len(set(gains))>1 else np.nan
            curves.append(dict(scope=f'D{D}_W{W}',d=d,Wmax=W,start_active_n=N,spearman=rho,last_minus_first=gains[-1]-gains[0],local_rebound_n=int((np.diff(gains)>1e-12).sum())))
    save('rq4_windows',windows);save('rq4_curves',curves)
    # Full10 runtime and selected token cohorts are independently recomputed from contributions.
    c=read('data/resources/runtime_contributions.csv');runtime=[]
    tasks=sorted(r.task_id.str.replace(r'_0$','',regex=True).unique())
    weights=np.random.default_rng(20260924).multinomial(len(tasks),np.full(len(tasks),1/len(tasks)),size=5000)
    for w,g in c.groupby('continuation_step'):
        N=g.execution_id.nunique();n=g.new_completion_contribution.sum();total=g.incremental_runtime_seconds.sum()
        per=g.groupby('task_id')[['initial_denominator_contribution','new_completion_contribution','incremental_runtime_seconds']].sum().reindex(tasks,fill_value=0)
        values=weights@per.to_numpy();gain=values[:,1]/values[:,0]
        cost=np.divide(values[:,2],values[:,1],out=np.full(5000,np.nan),where=values[:,1]>0)
        gl,gh=np.nanquantile(gain,[.025,.975]);cl,ch=np.nanquantile(cost,[.025,.975])
        runtime.append(dict(continuation_step=int(w),N=N,new_completion_n=int(n),gain=n/N,total_runtime=total,cost=total/n if n else np.nan,gain_ci_low=gl,gain_ci_high=gh,cost_ci_low=cl,cost_ci_high=ch,bootstrap_valid=int(np.isfinite(cost).sum())))
    save('rq4_runtime',runtime)
    c=read('data/resources/selected_resource_contributions.csv');tokens=[]
    for (metric,model,w),g in c.groupby(['metric','foundation_model','continuation_step']):
        N=g.execution_id.nunique();n=g.new_completion_contribution.sum();total=g.incremental_resource.sum()
        tokens.append(dict(metric=metric,model=model,w=w,N=N,new_completion_n=n,gain=n/N,total_resource=total,cost=total/n if n else np.nan))
    save('rq4_selected_resources',tokens)
    checks=[dict(check='1392 executions / 770 successes',passed=True),dict(check='572 diagnostic failures',passed=True),dict(check='17658 labels: 4194E / 13380N / 84U',passed=True),dict(check='final labels and raw GUI feature keys match one-to-one',passed=True),dict(check='error histories independently reconstructed from final labels',passed=True)]
    expected={0:(551,502,306),1:(753,242,88)}
    for row in summary:
        checks.append(dict(check=f"RQ3 success={row['success']} cohort/any-E/recurrence",passed=(row['n'],row['any_error_n'],row['recurrence_n'])==expected[row['success']]))
    expected_runtime={1:(.17138810198300283,191.67),5:(.03824362606232295,389.80),10:(.019830028328611898,520.30)}
    for row in runtime:
        checks.append(dict(check=f"runtime step {row['continuation_step']} fixed cohort",passed=row['N']==706))
        if row['continuation_step'] in expected_runtime:
            gain,cost=expected_runtime[row['continuation_step']]
            checks.append(dict(check=f"runtime step {row['continuation_step']} manuscript gain/cost",passed=np.isclose(row['gain'],gain) and np.isclose(row['cost'],cost,rtol=0,atol=.005)))
    checks.append(dict(check='83 continuation curves / 82 negative correlations / 78 negative endpoints',passed=len(curves)==83 and sum(x['spearman']<0 for x in curves)==82 and sum(x['last_minus_first']< -1e-12 for x in curves)==78))
    save('validation',checks)
    if not all(x['passed'] for x in checks):raise AssertionError(f'See {OUT}/validation.csv')
    print('Basic analyses and independent paper-result checks passed.',flush=True)

def models(boot=1000):
    import kernels as k
    import rq3_native as c
    c.B=boot;c.SEED=20261022
    f=rq3_frame()
    d=f[f.eligible_primary].copy();assert len(d)==16250
    rows=[]
    for name,terms in [('H0',[]),('H1',['fraction_z']),('H2',['fraction_z','previous_error']),('H3',['fraction_z','previous_error','excess_z'])]:
        print('Refitting history',name,flush=True)
        fit,covs,info=k.glm_fit(d,'y ~ '+' + '.join(terms+['position_z','C(task_id)','C(backbone)','C(configuration)']))
        rows+=k.glm_rows(fit,covs,info,name,terms,d)
    save('rq3_history_models',rows)
    features=['visual_stagnation','median_candidate_area','spatial_crowding','edge_density_clutter']
    for endpoint,mask,controls in [('onset',f.eligible_onset,['position_z','fraction_z']),('persistence',f.eligible_persistence,['position_z','fraction_z','excess_z'])]:
        for feature in features:
            print('Refitting GUI',endpoint,feature,flush=True)
            c.fit(f[mask],controls+[feature+'_z'],endpoint,'single_GUI',feature,boot=True)
    result=pd.DataFrame(c.MODELS);primary=result[result.primary.eq(True)].copy()
    for endpoint,part in primary.groupby('endpoint'):
        primary.loc[part.index,'q_value']=multipletests(part.p_value,method='fdr_bh')[1]
    save('rq3_gui_models',primary);save('rq3_gui_all_terms',result)
    jsonsave('rq3_bootstrap_failures.json',c.INVALID)
    h=pd.DataFrame(rows)
    for term,expected in [('fraction_z',1.67),('previous_error',1.80),('excess_z',2.25)]:
        actual=h[h.model.eq('H3')&h.cluster_unit.eq('task_id')&h.term.eq(term)].iloc[0].OR
        if not np.isclose(actual,expected,rtol=0,atol=.005):raise AssertionError('H3 manuscript discrepancy: '+term)
    assert len(primary)==8
    print('RQ3 history models and eight GUI models computed from final Excel labels.',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['basic','models','all']);p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--output-dir',type=Path,help='Generated analysis files; default is a directory in the system temporary folder')
    a=p.parse_args()
    global OUT
    if a.output_dir:
        os.environ['GUI_TRAJECTORY_WORKDIR']=str(a.output_dir.resolve())
        OUT=output_directory();OUT.mkdir(parents=True,exist_ok=True)
    if a.stage in ['basic','all']:
        basic();(OUT/'basic_inputs.sha256').write_text(input_digest())
    if a.stage in ['models','all']:
        models(a.bootstrap);(OUT/'model_inputs.sha256').write_text(input_digest()+':'+str(a.bootstrap))
    print('Generated analysis files:',OUT)
if __name__=='__main__':main()
