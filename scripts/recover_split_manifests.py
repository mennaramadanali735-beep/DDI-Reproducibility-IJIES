"""Verify archived split files and export ordered identifiers without raw molecules."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from build_split_manifests import record_hash

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SHA = '1ee9e7bdf33dbcce0341e74fd1b6855ec392a055eb506784928a09ed4364309f'
SPLIT_SHA = {
    'train.csv': '4ffbb76fe15bfc29b6c4736d2bae0f704c856cc0fca1834c3653f2762e6a52f4',
    'calibration.csv': '30a3e393a9d268555e89f894583a66ecdfd48a8a997d03ad512d9f49e6a7b050',
    'strict_test.csv': '4b1a4ab2c82775c8815153072812da2c15628915010970f4d4f58d5da89b7d4f',
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def identifiers(frame):
    keys = pd.Series([record_hash(row) for row in frame.to_dict('records')])
    return pd.DataFrame({'row_index':np.arange(len(frame)), 'record_sha256':keys,
        'occurrence_index':keys.groupby(keys,sort=False).cumcount(), 'label_id':frame.label_id.to_numpy()})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--side-train',type=Path,required=True)
    parser.add_argument('--scaffold',type=Path,required=True)
    args=parser.parse_args()
    assert sha(args.side_train)==TRAIN_SHA, 'Side-effect training file differs from archived identity'
    mapping=pd.read_csv(ROOT/'mappings/interaction_label_mapping.csv').set_index('interaction_label').class_id
    frame=pd.read_csv(args.side_train,keep_default_na=False)
    assert len(frame)==134249
    frame['label_id']=frame.Label.map(mapping)
    assert frame.label_id.notna().all()
    side=ROOT/'split/side_effects'
    identifiers(frame).to_csv(side/'train_ids.csv',index=False)
    meta=json.loads((side/'manifest.json').read_text())
    meta.update(molecular_train_manifest_available=True,train_molecular_manifest_sha256=sha(side/'train_ids.csv'),
        train_molecular_identity_scope='Exact ordered record hashes; same serialization as primary splits; within-split occurrence index')
    (side/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    folder=args.scaffold
    protocol=json.loads((folder/'protocol_metadata.json').read_text())
    for name,digest in protocol['saved_csv_sha256'].items():
        assert sha(folder/name)==digest,name
    for name,digest in SPLIT_SHA.items():assert sha(folder/name)==digest,name
    manifest=pd.read_csv(folder/'split_row_manifest.csv')
    dest=ROOT/'split/scaffold_nonchiral';dest.mkdir(exist_ok=True)
    report={'source_split_hashes':SPLIT_SHA,'scaffold_definition':protocol['scaffold_definition'],
        'original_rdkit_version':protocol['rdkit_version'],'scaffolds_recomputed_from_smiles':False,
        'audit_basis':'Hash-verified saved nonchiral scaffold columns; no new RDKit computation', 'splits':{}}
    sets={}
    for split,name in [('train','train.csv'),('calibration','calibration.csv'),('test','strict_test.csv')]:
        frame=pd.read_csv(folder/name,keep_default_na=False)
        ids=identifiers(frame)
        rows=manifest.loc[manifest.split.eq(split)].sort_values('saved_row')
        assert rows.saved_row.tolist()==list(range(len(frame)))
        ids['source_row']=rows.source_row.to_numpy()
        ids['source_file']=rows.source_file.to_numpy()
        for suffix in ['1','2']:
            ids['scaffold_'+suffix+'_sha256']=frame['scaffold_nonchiral_'+suffix].map(lambda v:hashlib.sha256(v.encode()).hexdigest())
        ids.to_csv(dest/f'{split}_ids.csv',index=False)
        sets[split]=set(frame.scaffold_nonchiral_1)|set(frame.scaffold_nonchiral_2)
        report['splits'][split]={'rows':len(frame),'classes':int(frame.label_id.nunique()),'manifest_sha256':sha(dest/f'{split}_ids.csv')}
    for split in ['train','calibration']:
        overlap=len(sets[split]&sets['test']);assert overlap==0
        report[split+'_test_saved_scaffold_overlap']=overlap
    assert report['splits']['test']['rows']==11142
    (dest/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
