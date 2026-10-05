"""Select thresholds using calibration probabilities and calibration targets only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def score(y,p):
    tp=np.sum(y&p,axis=0,dtype=np.int64);fp=np.sum(~y&p,axis=0,dtype=np.int64);fn=np.sum(y&~p,axis=0,dtype=np.int64)
    den=2*tp+fp+fn;per=np.divide(2*tp,den,out=np.zeros(len(tp),dtype=float),where=den!=0)
    micro=2*tp.sum()/den.sum() if den.sum() else 0.
    return 0.5*(micro+per.mean())

def select(y,prob):
    y=y.astype(bool);assert y.shape==prob.shape and y.shape[1]==263
    assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
    best=-1.;global_threshold=None
    for t in np.round(np.arange(.05,.851,.01),2):
        value=score(y,prob>=t)
        if value>best+1e-12:best=value;global_threshold=float(t)
    per=np.full(263,global_threshold,dtype=np.float32)
    for j in range(263):
        if y[:,j].sum()<3 or y[:,j].all():continue
        best_label=-1.
        for t in np.round(np.arange(.01,.951,.01),2):
            pred=prob[:,j]>=t;tp=np.sum(y[:,j]&pred);den=int(y[:,j].sum()+pred.sum());value=2*tp/den if den else 0.
            if value>best_label+1e-12:best_label=value;per[j]=t
    per_score=score(y,prob>=per)
    method='per_label' if per_score>best+1e-12 else 'global'
    return (per if method=='per_label' else np.full(263,global_threshold,dtype=np.float32)),{'selected_method':method,'global_threshold':global_threshold,'calibration_global_objective':best,'calibration_per_label_objective':per_score,'test_used_for_threshold_selection':False}

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=root/'results/threshold_reselection');args=parser.parse_args()
    a=np.load(root/'artifacts/side_effect_frozen_probabilities.npz',allow_pickle=False)
    thresholds,info=select(a['y_calibration'],a['calibration']);args.output.mkdir(exist_ok=True,parents=True);np.save(args.output/'thresholds.npy',thresholds)
    import pandas as pd
    expected=pd.read_csv(root/'configs/side_effect_thresholds_263.csv').threshold.to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(thresholds,expected)
    info['label_order_sha256']=hashlib.sha256((root/'mappings/side_effect_labels_263.csv').read_bytes()).hexdigest();info['historical_thresholds_reproduced_exactly']=True
    (args.output/'protocol.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
