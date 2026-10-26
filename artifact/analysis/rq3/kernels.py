"""Pure numerical kernels adapted from the audited RQ3 A/B and denominator code.

No legacy entrypoint or GUI module is imported. The conditional-likelihood DP
uses the same exact elementary-symmetric-polynomial estimator, vectorized over
trajectories to avoid a dependency on the historical optional numba runtime.
"""
import hashlib
import math
import warnings
import numpy as np
import pandas as pd
import patsy
from scipy import linalg, stats
from scipy.optimize import minimize
from scipy.special import expit
import statsmodels.api as sm
import statsmodels.formula.api as smf

def support(d):
    return hashlib.sha256('\n'.join(d.trajectory_id+'/'+d.source_step_index.astype(str)).encode()).hexdigest()

def z(s):
    sd=float(s.std(ddof=0))
    if sd<=0: raise ValueError('Zero-SD predictor')
    return (s-s.mean())/sd

def task_draws(tasks,B,seed):
    rng=np.random.default_rng(seed);T=len(tasks)
    return rng.multinomial(T,np.full(T,1/T),size=B)

def glm_fit(d,formula):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit=smf.glm(formula,data=d,family=sm.families.Binomial()).fit(maxiter=200,tol=1e-10)
    if not fit.converged:raise RuntimeError('Binomial logit did not converge')
    X=fit.model.exog;p=np.asarray(fit.fittedvalues)
    H=X.T@((p*(1-p))[:,None]*X)
    bread=np.linalg.pinv(H,rcond=1e-12);scores=X*(d.y.to_numpy()-p)[:,None]
    covs={}
    for unit in ['task_id','trajectory_id']:
        ids,codes=np.unique(d[unit].to_numpy(),return_inverse=True)
        sg=np.zeros((len(ids),X.shape[1]));np.add.at(sg,codes,scores)
        factor=len(ids)/(len(ids)-1)*(len(d)-1)/(len(d)-X.shape[1])
        covs[unit]=bread@(sg.T@sg)@bread*factor
    info=dict(n_observations=len(d),n_trajectories=d.trajectory_id.nunique(),n_tasks=d.task_id.nunique(),
        n_success_trajectories=d[d.final_success.eq(1)].trajectory_id.nunique(),
        n_failure_trajectories=d[d.final_success.eq(0)].trajectory_id.nunique(),
        n_cases=int(d.y.sum()),logLik=float(fit.llf),AIC=float(fit.aic),
        BIC=float(-2*fit.llf+len(fit.params)*np.log(len(d))),n_parameters=len(fit.params),
        support_hash=support(d),formula=formula,max_abs_score=float(np.max(np.abs(scores.sum(axis=0)))))
    return fit,covs,info

def glm_rows(fit,covs,info,name,focal,d):
    out=[]
    for unit,cov in covs.items():
        df=d[unit].nunique()-1;crit=stats.t.ppf(.975,df)
        for term in focal or ['controls_only']:
            r=dict(info,model=name,term=term,inference=unit+' clustered t; finite sample correction',cluster_unit=unit,cluster_df=df)
            if term!='controls_only':
                j=list(fit.params.index).index(term);b=float(fit.params.iloc[j]);se=math.sqrt(max(0,cov[j,j]))
                r.update(coefficient=b,SE=se,OR=np.exp(b),CI_low=np.exp(b-crit*se),CI_high=np.exp(b+crit*se),p_value=2*stats.t.sf(abs(b/se),df))
            out.append(r)
    return out

def perm_matrix(labels,positions,L,R,rng,blocked=False):
    valid=np.flatnonzero(labels>=0);a=np.tile(labels,(R,1))
    blocks=np.minimum(2,((positions[valid]-1)*3//L)) if blocked else np.zeros(len(valid),int)
    for b in np.unique(blocks):
        ix=valid[blocks==b]
        if len(ix):a[:,ix]=labels[ix][np.argsort(rng.random((R,len(ix))),axis=1)]
    # Frozen unknown slots and missing physical indices never become N.
    assert np.all((a==1).sum(axis=1)==(labels==1).sum())
    assert np.all((a==0).sum(axis=1)==(labels==0).sum())
    assert np.all(a[:,labels<0]==labels[labels<0])
    if blocked:
        for b in np.unique(blocks):
            ix=valid[blocks==b];assert np.all((a[:,ix]==1).sum(axis=1)==(labels[ix]==1).sum())
    mx=np.zeros(R,int);st=np.zeros(R,int);ee=np.zeros(R,int);en=np.zeros(R,int)
    for j in range(len(labels)):
        adjacent=j>0 and positions[j]==positions[j-1]+1
        st=np.where(a[:,j]==1,st+1 if adjacent else 1,0);mx=np.maximum(mx,st)
        if adjacent:
            origin=a[:,j-1]==1;ee+=origin&(a[:,j]==1);en+=origin&(a[:,j]==0)
    return mx,ee,en

def empirical(actual,null):
    a=np.asarray(null);a=a[np.isfinite(a)]
    if not np.isfinite(actual) or not len(a):return np.nan,np.nan
    hi=(1+np.sum(a>=actual-1e-12))/(len(a)+1);lo=(1+np.sum(a<=actual+1e-12))/(len(a)+1)
    return hi,min(1,2*min(hi,lo))

def ols_boot(d,response,B,seed):
    d=d.dropna(subset=[response]).copy();d['log_length_z']=z(np.log(d.execution_length));d['fraction_z']=z(d.error_fraction)
    formula=response+' ~ unsuccessful + log_length_z + fraction_z + C(task_id) + C(system)'
    yy,xx=patsy.dmatrices(formula,d,return_type='dataframe');names=list(xx.columns);j=names.index('unsuccessful')
    x=xx.to_numpy();y=yy.to_numpy().ravel()
    _,r,piv=linalg.qr(x,mode='economic',pivoting=True);rank=int(np.sum(abs(np.diag(r))>1e-9))
    keep=sorted(piv[:rank]);
    if j not in keep:raise RuntimeError('Focal outcome effect not identified')
    x=x[:,keep];j=keep.index(j);beta=linalg.lstsq(x,y,cond=1e-10,lapack_driver='gelsy')[0]
    tasks=sorted(d.task_id.unique());code=d.task_id.map({t:i for i,t in enumerate(tasks)}).to_numpy()
    cross=np.array([x[code==g].T@x[code==g] for g in range(len(tasks))]);xy=np.array([x[code==g].T@y[code==g] for g in range(len(tasks))])
    draws=[];invalid=[]
    for i,counts in enumerate(task_draws(tasks,B,seed)):
        mat=np.einsum('g,gij->ij',counts,cross);rhs=counts@xy
        # A rank loss in nuisance task columns is expected after task resampling.
        # The focal coefficient is valid only if every null-space vector has a
        # zero focal component. Do not assign minimum-norm unidentified effects.
        eig,V=linalg.eigh(mat);tol=max(float(eig[-1])*1e-9,1e-12)
        null=V[:,eig<=tol]
        if null.shape[1] and np.max(np.abs(null[j]))>1e-6:
            invalid.append(dict(repetition=i+1,reason='outcome coefficient not estimable in bootstrap draw'));continue
        positive=eig>tol;b=V[:,positive]@((V[:,positive].T@rhs)/eig[positive])
        if not np.all(np.isfinite(b)):
            invalid.append(dict(repetition=i+1,reason='nonfinite fit'));continue
        draws.append(float(b[j]))
    if len(draws)<500:raise RuntimeError(f'Only {len(draws)} identifiable bootstrap replicates')
    se=float(np.std(draws,ddof=1));lo,hi=np.quantile(draws,[.025,.975])
    result=dict(metric=response,formula=formula,n_trajectories=len(d),n_success=int(d.final_success.sum()),
        n_failure=int(d.unsuccessful.sum()),n_tasks=len(tasks),estimate_F_minus_S=float(beta[j]),
        CI_low=lo,CI_high=hi,bootstrap_SE=se,p_value_bootstrap_normal=2*stats.norm.sf(abs(beta[j]/se)),
        bootstrap_attempted=B,bootstrap_valid=len(draws),rank=rank,n_design_columns=len(xx.columns),
        inference='whole-task percentile bootstrap; normal approximation p')
    return result,draws,invalid

def prepare_cl(d,cols):
    eligible=d.dropna(subset=['y','trajectory_id']+cols).copy()
    v=eligible.groupby('trajectory_id').y.agg(['min','max']);ids=v[v['min']<v['max']].index
    q=eligible[eligible.trajectory_id.isin(ids)].sort_values('trajectory_id',kind='stable').copy()
    sizes=q.groupby('trajectory_id',sort=False).size().to_numpy();G=len(sizes);nmax=int(sizes.max());p=len(cols)
    X=np.zeros((G,nmax,p));Y=np.zeros((G,nmax));cursor=0
    for i,n in enumerate(sizes):
        X[i,:n]=q.iloc[cursor:cursor+n][cols].to_numpy(float);Y[i,:n]=q.iloc[cursor:cursor+n].y.to_numpy(float);cursor+=n
    return q,X,Y,sizes,v

def cl_parts(beta,X,Y,sizes):
    G,nmax,p=X.shape;original=Y.sum(axis=1).astype(int)
    complement=original>sizes-original;counts=np.minimum(original,sizes-original);mmax=int(counts.max())
    sign=np.where(complement,-1.,1.);Z=X*sign[:,None,None]
    valid=np.arange(nmax)[None,:]<sizes[:,None]
    cases=np.where(complement[:,None],1-Y,Y)*valid
    logits=Z@beta;num=(cases*logits).sum(axis=1);score_num=(cases[:,:,None]*Z).sum(axis=1)
    D=np.full((G,mmax+1),-np.inf);D[:,0]=0.
    Q=np.zeros((G,mmax+1,p));H=np.zeros((G,mmax+1,p,p))
    for i in range(nmax):
        for j in range(min(i+1,mmax),0,-1):
            ix=np.flatnonzero((sizes>i)&(counts>=j))
            if not len(ix):continue
            a=D[ix,j];b=D[ix,j-1]+logits[ix,i];den=np.logaddexp(a,b)
            wa=np.exp(a-den);wb=np.exp(b-den);right=Q[ix,j-1]+Z[ix,i];delta=Q[ix,j]-right
            H[ix,j]=wa[:,None,None]*H[ix,j]+wb[:,None,None]*H[ix,j-1]+(wa*wb)[:,None,None]*(delta[:,:,None]*delta[:,None,:])
            Q[ix,j]=wa[:,None]*Q[ix,j]+wb[:,None]*right;D[ix,j]=den
    ix=np.arange(G)
    return num-D[ix,counts],score_num-Q[ix,counts],H[ix,counts]

def conditional_fit(d,cols,name,legacy_interval=False):
    q,X,Y,sizes,v=prepare_cl(d,cols)
    def fun(b):
        ll,sc,H=cl_parts(b,X,Y,sizes);return -float(ll.sum()),-sc.sum(axis=0)
    opt=minimize(fun,np.zeros(len(cols)),jac=True,method='BFGS',options={'gtol':1e-7,'maxiter':200})
    ll,sc,H=cl_parts(opt.x,X,Y,sizes);grad=float(np.max(np.abs(sc.sum(axis=0))))
    if grad>1e-4:raise RuntimeError('Conditional fit failed score check: '+str(opt.message))
    bread=np.linalg.inv(H.sum(axis=0));meta=q.groupby('trajectory_id',sort=False).task_id.first()
    rows=[];covs={};N=len(q);p=len(cols)
    for unit in ['task_id','trajectory_id']:
        codes=pd.Categorical(meta if unit=='task_id' else meta.index).codes;G=int(codes.max()+1)
        scores=np.zeros((G,p));np.add.at(scores,codes,sc)
        factor=1 if legacy_interval else G/(G-1)*(N-1)/(N-p)
        cov=bread@(scores.T@scores)@bread*factor;covs[unit]=cov
        crit=1.96 if legacy_interval else stats.t.ppf(.975,G-1)
        for j,col in enumerate(cols):
            se=math.sqrt(max(0,cov[j,j]));b=float(opt.x[j])
            rows.append(dict(model=name,term=col,n_observations=N,n_trajectories=len(meta),n_tasks=q.task_id.nunique(),
                n_cases=int(q.y.sum()),eligible_steps=len(d),eligible_trajectories=d.trajectory_id.nunique(),
                excluded_all_E_trajectories=int((v['min'].eq(1)&v['max'].eq(1)).sum()),
                excluded_all_N_trajectories=int(v['max'].eq(0).sum()),excluded_constant_steps=len(d)-N,
                coefficient=b,SE=se,OR=np.exp(b),CI_low=np.exp(b-crit*se),CI_high=np.exp(b+crit*se),
                p_value=2*(stats.norm.sf(abs(b/se)) if legacy_interval else stats.t.sf(abs(b/se),G-1)),
                cluster_unit=unit,cluster_df=G-1,inference='legacy uncorrected sandwich normal' if legacy_interval else 'cluster score sandwich; finite sample t',
                logLik=float(ll.sum()),AIC=-2*ll.sum()+2*p,BIC=-2*ll.sum()+p*np.log(N),
                support_hash=support(q),max_abs_score=grad,formula='y ~ '+' + '.join(cols)+'; trajectory strata'))
    return rows,q,opt.x,covs
