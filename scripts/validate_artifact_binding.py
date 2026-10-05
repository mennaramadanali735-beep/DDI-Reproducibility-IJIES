"""Validate public label-order, threshold and saved-prediction binding."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    label_path=ROOT/'mappings/side_effect_labels_263.csv';threshold_path=ROOT/'configs/side_effect_thresholds_263.csv'
    labels=pd.read_csv(label_path);thresholds=pd.read_csv(threshold_path);protocol=json.loads((ROOT/'configs/side_effect_threshold_protocol.json').read_text())
    assert hashlib.sha256(label_path.read_bytes()).hexdigest()==protocol['label_order_sha256']
    assert hashlib.sha256(threshold_path.read_bytes()).hexdigest()==protocol['threshold_table_sha256']
    assert labels[['label_id','label']].equals(thresholds[['label_id','label']])
    p=np.load(ROOT/'artifacts/side_effect_frozen_probabilities.npz',allow_pickle=False)
    y=np.load(ROOT/'artifacts/side_effect_test_predictions.npz',allow_pickle=False)
    np.testing.assert_array_equal((p['test']>=thresholds.threshold.to_numpy(dtype=np.float32)).astype(np.uint8),y['ensemble'])
    print('Label order, threshold hashes and all saved Test predictions match')
