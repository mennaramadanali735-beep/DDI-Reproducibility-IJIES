"""Verify archived upstream source identities without executing training code."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('model',choices=['mhafr','autoddi']);p.add_argument('source',type=Path);args=p.parse_args()
    if args.model=='autoddi':
        hashes=json.loads((ROOT/'configs/autoddi/seed_42.json').read_text())['source_sha256']
        for name,digest in hashes.items():
            assert hashlib.sha256((args.source/name).read_bytes()).hexdigest()==digest,name
        print(f'Verified {len(hashes)} original AutoDDI source hashes')
    else:
        m=json.loads((ROOT/'evidence/mhafr/input_metadata.json').read_text())
        for name,key in [('drug_codes_chembl_freq_1500.txt','vocabulary_sha256'),('subword_units_map_chembl_freq_1500.csv','token_mapping_sha256')]:
            assert hashlib.sha256((args.source/'ESPF'/name).read_bytes()).hexdigest()==m[key]
        manifest=json.loads((ROOT/'configs/mhafr_source_hashes.json').read_text())
        for name,digest in manifest.items():assert hashlib.sha256((args.source/name).read_bytes()).hexdigest()==digest,name
        print(f'Verified {len(manifest)} MHAFR source and vocabulary hashes')

if __name__=='__main__':main()
