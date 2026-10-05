"""Verify packaged bytes against their manifest and report publication blockers."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    records=(ROOT/'SHA256SUMS.txt').read_text().splitlines()
    for line in records:
        expected,name=line.split('  ',1);path=ROOT/name
        assert path.is_file(),name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
    status=json.loads((ROOT/'results/reproduction_status.json').read_text())
    print(f'Verified {len(records)} packaged files')
    print('Table 9 complete:',status['table9_complete'])
    print('Consult docs/REMAINING_ITEMS.md before claiming full reproducibility.')
