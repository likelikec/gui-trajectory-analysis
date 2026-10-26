"""Portable entrypoint for the four paper implementations on AndroidWorld."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'runner'),str(ROOT/'agents')]

def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--model',choices=['qwen','gemini','gpt'],required=True)
    p.add_argument('--mode',choices=['p1','p2','p3','p4'],required=True)
    p.add_argument('--tasks',help='Comma-separated task template names; default: the 116 paper tasks')
    p.add_argument('--defaults',type=Path,help='Override the portable model configuration')
    p.add_argument('--dry-run',action='store_true',help='Resolve configuration without a model API or device call')
    args,extra=p.parse_known_args()
    tasks=args.tasks or ','.join(json.loads((ROOT/'configs/tasks.json').read_text()))
    defaults=args.defaults.resolve() if args.defaults else ROOT/'configs'/(args.model+'.yaml')
    sys.argv=[str(ROOT/'runner/study/run_mode.py'),'--mode',args.mode,'--benchmark','androidworld','--defaults',str(defaults),'--tasks',tasks,*extra]
    if args.dry_run:sys.argv.append('--dry-run')
    from study.run_mode import main as run
    run()

if __name__=='__main__':main()
