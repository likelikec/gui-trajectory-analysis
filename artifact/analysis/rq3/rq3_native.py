"""Numerical RQ3 kernel extracted from the preserved current implementation.
Only native loading/platform flags and output locations adapted. No legacy entrypoint imported.
"""
import ctypes, hashlib, json, math, subprocess, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from scipy import stats
from kernels import prepare_cl,cl_parts,support
HERE=Path(__file__).resolve().parent
from workspace import output_directory
CACHE=output_directory()/'bootstrap'
CACHE.mkdir(parents=True,exist_ok=True)
SEED=20261022
B=1000
MODELS=[];SENS=[];COV=[];INVALID=[]
def log(s):print(s,flush=True)
def native():
    src=HERE/'conditional.cpp'
    lib=CACHE/('conditional.dylib' if sys.platform=='darwin' else 'conditional.so')
    if not lib.exists() or lib.stat().st_mtime<src.stat().st_mtime:
        flags=['-dynamiclib'] if sys.platform=='darwin' else ['-shared','-fPIC']
        subprocess.run(['c++','-O3','-std=c++17',*flags,str(src),'-o',str(lib)],check=True)
    api=ctypes.CDLL(str(lib)).evaluate
    ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    ints=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    api.argtypes=[ctypes.c_int]*3+[ptr,ptr,ints,ptr,ptr,ptr];api.restype=None
    def evaluate(b,X,Y,sizes,w):
        p=X.shape[2];out=np.empty(1+p+p*p)
        api(X.shape[0],X.shape[1],p,X,Y,sizes,np.ascontiguousarray(b),np.ascontiguousarray(w),out)
        return out[0],out[1:1+p],out[1+p:].reshape(p,p)
    return evaluate
EVAL=native()
def fit_weighted(X,Y,sizes,w,start):
 b=start.copy()
 for it in range(60):
  ll,sc,H=EVAL(b,X,Y,sizes,w)
  if not np.isfinite(ll) or not np.isfinite(H).all():raise ValueError('nonfinite likelihood')
  ev=np.linalg.eigvalsh(H)
  if ev[0]<=1e-10 or ev[-1]/ev[0]>1e12:raise ValueError('unidentified/ill-conditioned information')
  if np.max(abs(sc))<1e-5:
   if np.max(abs(b))>30:raise ValueError('unbounded coefficient / possible separation')
   return b
  delta=np.linalg.solve(H,sc);alpha=1
  while alpha>1e-8:
   cand=b+alpha*delta;ll2=EVAL(cand,X,Y,sizes,w)[0]
   if np.isfinite(ll2) and ll2>=ll-1e-9:break
   alpha*=.5
  if alpha<=1e-8:raise ValueError('line search failed')
  b=cand
 raise ValueError('did not converge')

def fit(d,cols,endpoint,name,feature,boot=False,sensitivity=False,replication=False):
 eligible=d.dropna(subset=cols+['y']).copy()
 q,X,Y,sizes,v=prepare_cl(eligible,cols);sizes=np.ascontiguousarray(sizes,dtype=np.int64)
 X=np.ascontiguousarray(X);Y=np.ascontiguousarray(Y)
 if not len(q):raise ValueError('No informative trajectories')
 start=np.zeros(len(cols));beta=fit_weighted(X,Y,sizes,np.ones(len(sizes)),start)
 # Once per model, use original DP for score-cluster covariance.
 ll,sc,H=cl_parts(beta,X,Y,sizes);assert np.max(abs(sc.sum(axis=0)))<1e-4
 assert np.isclose(ll.sum(),EVAL(beta,X,Y,sizes,np.ones(len(sizes)))[0],atol=1e-8)
 bread=np.linalg.inv(H.sum(axis=0));meta=q.groupby('trajectory_id',sort=False).task_id.first()
 info=dict(endpoint=endpoint,model=name,feature=feature,eligible_steps=len(eligible),eligible_trajectories=eligible.trajectory_id.nunique(),n_observations=len(q),n_trajectories=len(meta),n_tasks=q.task_id.nunique(),n_cases=int(q.y.sum()),excluded_all_E_trajectories=int((v['min'].eq(1)&v['max'].eq(1)).sum()),excluded_all_N_trajectories=int(v['max'].eq(0).sum()),excluded_constant_steps=len(eligible)-len(q),logLik=float(ll.sum()),AIC=-2*float(ll.sum())+2*len(cols),BIC=-2*float(ll.sum())+len(cols)*np.log(len(q)),n_parameters=len(cols),support_hash=support(q),formula='y ~ '+' + '.join(cols)+'; strata=trajectory_id',max_abs_score=float(np.max(abs(sc.sum(axis=0)))))
 bootci=None;valid=0
 if boot:
  file=CACHE/(endpoint+'_'+name+'_'+feature+'.npz');fingerprint=hashlib.sha256((support(q)+str(cols)+str(beta)+str(SEED)+str(B)+hashlib.sha256(X.tobytes()+Y.tobytes()).hexdigest()).encode()).hexdigest()
  tasks=sorted(q.task_id.unique());ti={t:i for i,t in enumerate(tasks)};codes=np.array([ti[x] for x in meta]);rng=np.random.default_rng(SEED);weights=rng.multinomial(len(tasks),np.full(len(tasks),1/len(tasks)),size=B)
  if file.exists():
   stored=np.load(file)
   if str(stored['fingerprint'])!=fingerprint:stored.close();file.unlink()
   else:stored.close()
  if file.exists():
   stored=np.load(file);assert str(stored['fingerprint'])==fingerprint;draws=stored['draws'];invalid=json.loads(str(stored['invalid']))
  else:
   def run(item):
    index,w=item
    try:return index,fit_weighted(X,Y,sizes,w[codes].astype(float),beta),None
    except (ValueError,np.linalg.LinAlgError) as e:return index,None,str(e)
   draws=[];invalid=[]
   with ThreadPoolExecutor(max_workers=4) as pool:
    for index,b,reason in pool.map(run,enumerate(weights)):
     if reason:invalid.append(dict(repetition=index+1,reason=reason))
     else:draws.append(b)
     if (index+1)%250==0:log(f'{endpoint} {name}/{feature}: bootstrap {index+1}/{B}; invalid={len(invalid)}')
   draws=np.array(draws);np.savez_compressed(file,draws=draws,fingerprint=fingerprint,invalid=json.dumps(invalid))
  valid=len(draws);bootci=np.quantile(draws,[.025,.975],axis=0) if valid>=2 else np.full((2,len(cols)),np.nan)
  INVALID.extend(dict(endpoint=endpoint,model=name,feature=feature,**r) for r in invalid)
 rows=[];covs={}
 for unit in ['task_id','trajectory_id']:
  codes=pd.Categorical(meta if unit=='task_id' else meta.index).codes;G=int(codes.max()+1);scores=np.zeros((G,len(cols)));np.add.at(scores,codes,sc)
  cov=bread@(scores.T@scores)@bread*(G/(G-1))*((len(q)-1)/(len(q)-len(cols)));covs[unit]=cov
  for j,col in enumerate(cols):
   se=math.sqrt(max(0,cov[j,j]));crit=stats.t.ppf(.975,G-1)
   r=dict(info,term=col,cluster_unit=unit,cluster_df=G-1,coefficient=beta[j],SE=se,OR=np.exp(beta[j]),CI_low=np.exp(beta[j]-crit*se),CI_high=np.exp(beta[j]+crit*se),p_value=2*stats.t.sf(abs(beta[j]/se),G-1),inference='cluster-score sandwich; finite-sample t',primary=False)
   if boot and unit=='task_id':
    r.update(cluster_CI_low=r['CI_low'],cluster_CI_high=r['CI_high'],CI_low=np.exp(bootci[0,j]),CI_high=np.exp(bootci[1,j]),bootstrap_attempted=B,bootstrap_valid=valid,bootstrap_unstable=valid<950,inference='whole-task percentile OR CI; p from task-cluster finite-sample t',primary=name=='single_GUI' and col==feature+'_z')
   rows.append(r)
 if not replication:(SENS if sensitivity else MODELS).extend(rows)
 COV.append(dict(info,record_type='fit_support',n_success_steps=int(q.final_success.sum()),n_failure_steps=int((1-q.final_success).sum()),n_success_trajectories=q[q.final_success.eq(1)].trajectory_id.nunique(),n_failure_trajectories=q[q.final_success.eq(0)].trajectory_id.nunique()))
 return dict(rows=rows,info=info,beta=beta,covs=covs,cols=cols,q=q)
