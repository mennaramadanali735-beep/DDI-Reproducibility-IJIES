"""Launch archived final AutoDDI training or evaluation with verified source files."""
import argparse
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,43,44],required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);args=p.parse_args()
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_upstream.py'),'autoddi',str(args.source)],check=True)
    args.run_root.mkdir(parents=True,exist_ok=True)
    script=ROOT/f'scripts/autoddi_final/seed_{args.seed}/train_final_seed{args.seed}.py'
    subprocess.run([sys.executable,str(script),str(args.inputs),str(args.source),str(args.run_root)],check=True)
