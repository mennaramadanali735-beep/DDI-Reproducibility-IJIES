"""Regenerate exact common-record integrated metrics and four-category counts."""
from pathlib import Path
import numpy as np
from reproduce_tables_8_to_13 import ROOT, integrated
if __name__=='__main__':
    out=ROOT/'results';out.mkdir(exist_ok=True)
    data=np.load(ROOT/'artifacts/side_effect_test_predictions.npz',allow_pickle=False)
    print(integrated(out,data))
