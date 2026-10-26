"""Check final workbook schemas, dataset identities, case links and package contents."""
import ast
import json
import re
import sys
import zipfile
from pathlib import Path
import pandas as pd

from reproduce import ROOT,executions,read,rq2_labels

def main():
    package=ROOT.parent
    runs=executions()
    assert len(runs)==1392 and runs.success.sum()==770
    assert runs.groupby(['model','implementation']).size().eq(116).all()
    q=rq2_labels();assert len(q)==572 and q.execution_id.nunique()==572
    assert set(q.execution_id)<=set(runs[runs.success.eq(0)].execution_id)
    codes={f+str(k) for f,n in [('R',3),('D',7),('E',3),('M',2)] for k in range(1,n+1)}
    assert set(q.dominant_code)<=codes
    assert set(q.cooccurring_code.dropna())<=codes
    assert not (q.dominant_code==q.cooccurring_code).any()
    schemas={
        'rq2_annotations.xlsx':['execution_id','dominant_code','cooccurring_code'],
        'rq3_annotations.xlsx':['trajectory_id','source_step_index','label'],
    }
    for name,fields in schemas.items():
        frame=read('data/'+name);assert list(frame)==fields,(name,list(frame))
    labels=read('data/rq3_annotations.xlsx')
    assert len(labels)==17658 and labels.trajectory_id.nunique()==1304
    assert labels.label.value_counts().to_dict()=={'N':13380,'E':4194,'U':84}
    assert set(labels.trajectory_id)<=set(runs.execution_id)
    keys=['trajectory_id','source_step_index']
    assert not labels.duplicated(keys).any() and labels.source_step_index.ge(1).all()
    features=read('data/gui_features/rq3_steps.csv')
    assert list(features)==keys+['visual_stagnation','median_candidate_area','spatial_crowding','edge_density_clutter']
    matched=labels.merge(features,on=keys,validate='one_to_one',how='outer',indicator=True)
    assert len(matched)==len(labels) and matched._merge.eq('both').all()
    for name in ['execution_results.xlsx','rq2_annotations.xlsx','rq3_annotations.xlsx']:
        with zipfile.ZipFile(ROOT/'data'/name) as workbook:
            assert workbook.testzip() is None
            assert not any(n.startswith('xl/externalLinks/') for n in workbook.namelist())
    cases=json.loads((ROOT/'cases/index.json').read_text())
    assert len(cases)==16
    assert len({case['model']+'/'+case['implementation'] for case in cases})==12
    for case in cases:
        steps=json.loads((ROOT/case['timeline']).read_text())
        for step in steps:
            assert step['label'] in ['E','N','U']
            for name in [*step['images'].values(),*([step['record']] if step.get('record') else [])]:
                path=(ROOT/name).resolve()
                assert path.is_relative_to(ROOT) and path.is_file(),name
    secret=re.compile(rb'(?:sk-[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|AIza[A-Za-z0-9_-]{30,})')
    scan_roots=[ROOT]
    if (package/'pipline').is_dir():scan_roots.append(package/'pipline')
    for path in (item for folder in scan_roots for item in folder.rglob('*')):
        if any(part in ['.git','__pycache__','outputs','.cache'] for part in path.relative_to(package).parts):continue
        assert not path.is_symlink(),path
        if path.is_dir():
            assert path.name not in ['results','provenance'],path
            continue
        if path.suffix in ['.py','.js','.json','.yaml','.yml','.md','.txt','.csv']:
            assert not secret.search(path.read_bytes()),path
        if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    if (ROOT/'website/data.js').exists():
        text=(ROOT/'website/data.js').read_text();payload=json.loads(text.split('=',1)[1].rstrip(';\n'))
        assert len(payload['runs'])==1392 and len(payload['rq2'])==572
        assert sum(row['success'] for row in payload['runs'])==770
        assert len(payload['downloads'])==3
        for download in payload['downloads']:assert (ROOT/'website'/download['url']).resolve().is_file()
    print('PASS final Excel schemas and totals; GUI keys; selected cases; standalone artifact; credential scan; Python syntax.')

if __name__=='__main__':main()
