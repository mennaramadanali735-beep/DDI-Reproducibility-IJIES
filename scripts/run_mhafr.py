"""Launch recovered MHAFR stages with explicit inputs and seed selection."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pretrain','embeddings','classify']);p.add_argument('--seed',type=int,choices=[42,43,44],required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--source',type=Path,default=ROOT/'third_party/MHAFR-DDI');args=p.parse_args()
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_upstream.py'),'mhafr',str(args.source)],check=True)
    base=ROOT/'scripts/recovered_cold_start/mhafr_pretraining_v1';args.run_root.mkdir(parents=True,exist_ok=True)
    if args.stage=='pretrain':
        assert (args.run_root/'molecular_pretrain_split.csv').is_file()
        cmd=[base/f'pretrain_seed{args.seed}.py',args.source,args.features,args.run_root]
    elif args.stage=='embeddings':cmd=[base/'extract_embeddings.py',args.source,args.features,args.run_root/f'seed_{args.seed}']
    else:cmd=[base/f'classify_seed{args.seed}.py',args.features,args.run_root/f'seed_{args.seed}']
    subprocess.run([sys.executable,*map(str,cmd)],check=True)
