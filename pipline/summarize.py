"""Summarize recorded new runs with the original reporting writer; no model/device calls."""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'runner'),str(ROOT/'agents')]

class LiteralTranslator:
    def translate_text(self,text):return text

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--run-names',help='Optional comma-separated run names; otherwise include every recorded run')
    args=p.parse_args()
    source=args.output_root.resolve();target=args.output.resolve()
    if target.is_relative_to(ROOT.parent/'artifact/data'):
        raise SystemExit('New run summaries must not overwrite the packaged final analysis data.')
    selected=set(args.run_names.split(',')) if args.run_names else None
    records=[]
    for file in sorted(source.glob('androidworld/p*/*/run_report.json')):
        if selected is not None and file.parent.name not in selected:continue
        records.extend(json.loads(file.read_text()).get('records',[]))
    if not records:raise SystemExit('No recorded run_report.json records found in the selected runs.')
    from study import reporting
    target.parent.mkdir(parents=True,exist_ok=True)
    with patch.object(reporting,'ChineseTranslator',LiteralTranslator):
        reporting.write_records_to_excel(target,records)
    print(f'Exported {len(records)} recorded executions to {target}; no automatic rerun selection or human correction applied.')

if __name__=='__main__':main()
