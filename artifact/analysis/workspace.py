"""Generated analysis files live outside the distributed research inputs."""
import hashlib
import os
import tempfile
from pathlib import Path

ARTIFACT=Path(__file__).resolve().parents[1]

def output_directory():
    explicit=os.environ.get('GUI_TRAJECTORY_WORKDIR')
    if explicit:return Path(explicit).expanduser().resolve()
    identity=hashlib.sha256(str(ARTIFACT).encode()).hexdigest()[:10]
    return Path(tempfile.gettempdir())/('gui-trajectory-analysis-'+identity)

def input_digest():
    """Invalidate derived files whenever canonical data or numerical code changes."""
    sources=sorted((ARTIFACT/'data').rglob('*.xlsx'))+sorted((ARTIFACT/'data').rglob('*.csv'))
    sources += [ARTIFACT/'analysis/reproduce.py',ARTIFACT/'analysis/rq3/kernels.py',ARTIFACT/'analysis/rq3/rq3_native.py',ARTIFACT/'analysis/rq3/conditional.cpp']
    digest=hashlib.sha256()
    for source in sources:
        digest.update(str(source.relative_to(ARTIFACT)).encode())
        digest.update(b'\0');digest.update(source.read_bytes());digest.update(b'\0')
    return digest.hexdigest()
