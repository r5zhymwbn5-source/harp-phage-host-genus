"""Regenerate the integrity manifest, excluding Git and local test output."""
from pathlib import Path
import hashlib

root = Path(__file__).resolve().parents[1]
files = [p for p in root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS'
         and not set(p.relative_to(root).parts) & {'.git', 'scratch', '__pycache__', '.pytest_cache'}
         and p.suffix not in {'.pyc', '.pyo'}]
with (root/'SHA256SUMS').open('w', encoding='utf8', newline='\n') as out:
    for path in sorted(files):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):
                h.update(block)
        out.write(f'{h.hexdigest()}  {path.relative_to(root).as_posix()}\n')
print('recorded',len(files),'files')
