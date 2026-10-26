"""Package the two-component repository; generated analyses are not distributed."""
import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT.parent/(ROOT.name+'.zip'))
    args=p.parse_args()
    assert sorted(item.name for item in ROOT.iterdir() if item.is_dir() and not item.name.startswith('.'))==['artifact','pipline']
    subprocess.run([sys.executable,str(Path(__file__).with_name('verify.py'))],check=True)
    destination=args.output.resolve()
    if destination.is_relative_to(ROOT):raise SystemExit('Write the download archive outside the source directory.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    skip={'.git','__pycache__','.cache','outputs','.secrets','node_modules','.venv','.venv-pipline','.venv-artifact'}
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for file in sorted(ROOT.rglob('*')):
            relative=file.relative_to(ROOT)
            if not file.is_file() or any(part in skip for part in relative.parts) or file.name=='.DS_Store' or file.suffix in ['.pyc','.zip'] or file.name.endswith('.inspect.ndjson'):continue
            assert not file.is_symlink()
            assert not any(part in ['provenance','results'] for part in relative.parts)
            info=zipfile.ZipInfo(str(Path('gui-trajectory-analysis')/relative),date_time=(2026,10,26,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
            archive.writestr(info,file.read_bytes())
    with zipfile.ZipFile(destination) as archive:assert archive.testzip() is None
    print('Created',destination,f'({destination.stat().st_size/1048576:.1f} MiB)')

if __name__=='__main__':main()
