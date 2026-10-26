"""Build the bilingual explorer directly from canonical Excel data and computed analyses."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from reproduce import ROOT,executions,read,rq2_labels
from workspace import output_directory,input_digest

def records(frame):return json.loads(frame.to_json(orient='records',force_ascii=False))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-dir',type=Path,help='Existing output from analysis/run.py all')
    args=p.parse_args()
    if args.analysis_dir:os.environ['GUI_TRAJECTORY_WORKDIR']=str(args.analysis_dir.resolve())
    work=output_directory()
    current=input_digest()
    signatures_current=all((work/name).is_file() and (work/name).read_text()==expected for name,expected in [('basic_inputs.sha256',current),('model_inputs.sha256',current+':1000')])
    if not signatures_current or not all((work/name).is_file() for name in ['rq3_history_models.csv','rq3_gui_models.csv','rq4_runtime.csv','rq1_task_coverage.csv']):
        subprocess.run([sys.executable,str(Path(__file__).with_name('run.py')),'all','--output-dir',str(work)],check=True)
    def computed(name):
        import pandas as pd
        return pd.read_csv(work/(name+'.csv'),float_precision='round_trip')
    r=executions();cases=json.loads((ROOT/'cases/index.json').read_text())
    for case in cases:
        case['timeline']=json.loads((ROOT/case['timeline']).read_text())
        for step in case['timeline']:
            step['images']={key:'../'+value for key,value in step['images'].items()}
            if step.get('record'):step['record']='../'+step['record']
    history=computed('rq3_history_models');history=history[history.model.eq('H3')&history.cluster_unit.eq('task_id')]
    tokens=computed('rq4_selected_resources');tokens=tokens[tokens.metric.eq('tokens')]
    downloads=[]
    for name,en,zh,description_en,description_zh in [
        ('execution_results.xlsx','Execution summary','执行结果汇总','Final outcomes and interaction lengths for 1,392 executions.','1,392 条执行的最终终态与交互步数。'),
        ('rq2_annotations.xlsx','RQ2 final annotations','RQ2 最终标注','572 final dominant-cause labels, with optional co-occurring causes.','572 条最终主导原因标注及可选伴随原因。'),
        ('rq3_annotations.xlsx','RQ3 final annotations','RQ3 最终标注','17,658 final E/N/U step labels.','17,658 步最终 E/N/U 标签。')]:
        file=ROOT/'data'/name
        downloads.append(dict(kind='EXCEL',filename=name,title_en=en,title_zh=zh,description_en=description_en,description_zh=description_zh,bytes=file.stat().st_size,url='../data/'+name))
    payload=dict(version=(ROOT/'VERSION').read_text().strip(),runs=records(r[['execution_id','model','implementation','task_id','task_difficulty','optimal_steps','step_count','success']]),
        taskCoverage=r.groupby('task_id').success.sum().to_dict(),rq2=records(rq2_labels()[['model','mode','task_id','dominant_code','cooccurring_code']]),
        causes=records(computed('rq2_causes').sort_values('count',ascending=False)),cooccurrence=records(computed('rq2_cooccurrence')),
        errorSummary=records(computed('rq3_summary')),history=records(history),gui=records(computed('rq3_gui_models')),
        windows=records(computed('rq4_windows')),runtime=records(computed('rq4_runtime')),tokenCosts=records(tokens),cases=cases,downloads=downloads)
    (ROOT/'website/data.js').write_text('window.ARTIFACT_DATA='+json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':'))+';\n')
    print('Built artifact/website/data.js from the final Excel workbooks.')

if __name__=='__main__':main()
