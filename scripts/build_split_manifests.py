"""Export ordered, content-addressed split identifiers without molecular strings."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def record_hash(row):
    payload = json.dumps([row['Drug1'], row['Drug2'], row['Label'], int(row['label_id'])], ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def export(source, output, expected):
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'format': 'ordered-record-sha256-v1', 'serialization': 'UTF-8 compact JSON [Drug1, Drug2, Label, integer label_id]; exact strings; no molecular normalization', 'occurrence_scope': 'within each split in saved row order', 'splits': {}}
    for split in ['train', 'calibration', 'test']:
        path = source / f'clean_{split}_split.csv'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected[split]:
            raise ValueError(f'{split}: original split file SHA-256 mismatch')
        seen = Counter()
        dest = output / f'{split}_ids.csv'
        with path.open(newline='', encoding='utf-8-sig') as inp, dest.open('w', newline='', encoding='utf-8') as out:
            writer = csv.writer(out); writer.writerow(['row_index', 'record_sha256', 'occurrence_index', 'class_id'])
            count = 0
            for i, row in enumerate(csv.DictReader(inp)):
                key = record_hash(row)
                writer.writerow([i, key, seen[key], int(row['label_id'])]); seen[key] += 1; count += 1
        manifest['splits'][split] = {'rows': count, 'source_file_sha256': digest, 'manifest_sha256': hashlib.sha256(dest.read_bytes()).hexdigest(), 'duplicate_occurrences': sum(v-1 for v in seen.values())}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--identity', type=Path, default=Path(__file__).resolve().parents[1]/'configs/primary_split_identity.json')
    args = parser.parse_args(); export(args.source, args.output, json.loads(args.identity.read_text())['split_sha256'])
