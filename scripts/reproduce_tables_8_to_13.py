"""Recompute manuscript tables from released predictions and archived run outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss

ROOT = Path(__file__).resolve().parents[1]

def interaction_metrics(y, pred):
    return dict(accuracy=accuracy_score(y,pred), macro_f1=f1_score(y,pred,labels=np.arange(84),average='macro',zero_division=0), weighted_f1=f1_score(y,pred,labels=np.arange(84),average='weighted',zero_division=0))

def side_metrics(y,pred):
    return dict(exact_match=float(np.all(y==pred,axis=1).mean()),micro_f1=f1_score(y,pred,average='micro',zero_division=0),macro_f1=f1_score(y,pred,average='macro',zero_division=0),weighted_f1=f1_score(y,pred,average='weighted',zero_division=0),hamming_loss=hamming_loss(y,pred))

def aggregate(frame,group):
    rows=[]
    for key,d in frame.groupby(group,sort=False):
        assert sorted(d.seed.tolist())==[42,43,44]
        row={group:key,'n':len(d)}
        for metric in ['accuracy','macro_f1','weighted_f1']:
            row[metric+'_mean']=d[metric].mean();row[metric+'_sd']=d[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)

def integrated(out,side):
    primary=pd.read_csv(ROOT/'split/primary_interaction/test_ids.csv')
    metadata=pd.read_csv(ROOT/'split/side_effects/test_ids.csv')
    key=['record_sha256','occurrence_index']
    assert not primary.duplicated(key).any() and not metadata.duplicated(key).any()
    eligible=metadata.loc[metadata.candidate_count.eq(1)]
    matched=eligible.merge(primary[key],on=key,how='inner',validate='one_to_one').sort_values('row_index')
    predictions=pd.read_csv(ROOT/'artifacts/integrated_interaction_predictions.csv').sort_values('side_test_row')
    assert matched.row_index.tolist()==predictions.side_test_row.tolist()
    assert matched.record_sha256.tolist()==predictions.record_sha256.tolist()
    assert len(predictions)==4390
    index=predictions.side_test_row.to_numpy()
    y=side['y_true'][index];p=side['ensemble'][index]
    interaction_ok=predictions.y_true.to_numpy()==predictions.y_pred.to_numpy()
    side_ok=(y==p).all(axis=1)
    lookup=np.zeros((84,263),dtype=np.uint8)
    for row in pd.read_csv(ROOT/'mappings/train_class_to_target_ids.csv',keep_default_na=False).itertuples():
        ids=[int(x) for x in row.Target_Vector_String.split('|') if x]
        lookup[row.class_id,ids]=1
    assert np.array_equal(lookup[predictions.y_true.to_numpy()],y)
    lookup_ok=(lookup[predictions.y_pred.to_numpy()]==y).all(axis=1)
    rows=[]
    for name,correct in [('GCN concat 44 + side ensemble',side_ok),('Same GCN + Train-only lookup',lookup_ok)]:
        rows.append({'model':name,'n':len(index),'interaction_accuracy':interaction_ok.mean(),'side_exact_accuracy':correct.mean(),'strict_accuracy':(interaction_ok&correct).mean()})
    pd.DataFrame(rows).to_csv(out/'table11.csv',index=False)
    categories=[('Both correct',interaction_ok&side_ok),('Only interaction wrong',~interaction_ok&side_ok),('Only side-effect wrong',interaction_ok&~side_ok),('Both wrong',~interaction_ok&~side_ok)]
    counts=[{'category':n,'count':int(mask.sum()),'percent':100*mask.mean()} for n,mask in categories]
    assert [d['count'] for d in counts]==[3216,56,905,213]
    pd.DataFrame(counts).to_csv(out/'table12.csv',index=False)
    pd.DataFrame({'side_test_row':index,'interaction_correct':interaction_ok,'side_exact_correct':side_ok,'lookup_exact_correct':lookup_ok}).to_csv(out/'integrated_row_audit.csv',index=False)
    return rows[0]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results');parser.add_argument('--table9-extra',type=Path)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    data=np.load(ROOT/'artifacts/interaction_test_predictions.npz',allow_pickle=False);rows=[]
    for model,prefix in [('Morgan LightGBM','morgan'),('Weighted GCN (historical V2 cache)','gcn')]:
        for seed in [42,43,44]:rows.append({'model':model,'seed':seed,**interaction_metrics(data['y_true'],data[f'{prefix}_seed_{seed}'])})
    per_seed=pd.DataFrame(rows);per_seed.to_csv(out/'table8_runs.csv',index=False);t8=aggregate(per_seed,'model');t8.to_csv(out/'table8.csv',index=False)
    historical=pd.read_csv(ROOT/'artifacts/historical_ablation_run_metrics.csv').rename(columns={'test_accuracy':'accuracy','test_macro_f1':'macro_f1','test_weighted_f1':'weighted_f1'})
    historical=historical[['setting','seed','accuracy','macro_f1','weighted_f1']]
    weighted=historical[historical.setting.eq('weighted')].copy();weighted['setting']='four_way';historical=pd.concat([historical,weighted])
    required={'morgan_only','rdkit_only','morgan_rdkit','checkpoint_macro_f1','checkpoint_accuracy','checkpoint_weighted_f1'}
    checkpoint=pd.read_csv(ROOT/'artifacts/all_seed_checkpoint_selection_results.csv')
    assert set(checkpoint.criterion)=={'accuracy','macro_f1','weighted_f1'}
    checkpoint['setting']='checkpoint_'+checkpoint.criterion
    checkpoint=checkpoint.rename(columns={'test_accuracy':'accuracy','test_macro84':'macro_f1','test_weighted_f1':'weighted_f1'})
    historical=pd.concat([historical,checkpoint[historical.columns]],ignore_index=True)
    for mode in ['morgan_only','rdkit_only','morgan_rdkit']:
        feature=pd.read_csv(ROOT/'artifacts/feature_ablation'/f'{mode}_all_seed_results.csv')
        feature['setting']=mode
        feature=feature.rename(columns={'test_accuracy':'accuracy','test_macro84':'macro_f1','test_weighted_f1':'weighted_f1'})
        historical=pd.concat([historical,feature[historical.columns]],ignore_index=True)
    if args.table9_extra:
        extra=pd.read_csv(args.table9_extra)
        assert set(extra.setting).issubset(required-set(historical.setting)), 'Only missing settings may be supplied'
        historical=pd.concat([historical,extra[historical.columns]],ignore_index=True)
    assert not historical.duplicated(['setting','seed']).any()
    metric_columns=['accuracy','macro_f1','weighted_f1']
    assert historical[metric_columns].notna().all().all()
    assert historical[metric_columns].ge(0).all().all() and historical[metric_columns].le(1).all().all()
    historical.to_csv(out/'table9_runs.csv',index=False)
    missing=sorted(required-set(historical.setting))
    (out/('table9.csv' if missing else 'table9_PARTIAL.csv')).unlink(missing_ok=True)
    t9=aggregate(historical,'setting');t9.to_csv(out/('table9_PARTIAL.csv' if missing else 'table9.csv'),index=False)
    side=np.load(ROOT/'artifacts/side_effect_test_predictions.npz',allow_pickle=False)
    individual=pd.DataFrame([{'seed':seed,**side_metrics(side['y_true'],side[f'seed_{seed}'])} for seed in [42,43,44]])
    ensemble=side_metrics(side['y_true'],side['ensemble']);individual.to_csv(out/'table10_runs.csv',index=False)
    pd.DataFrame([{'metric':k,'individual_mean':individual[k].mean(),'individual_sample_sd':individual[k].std(ddof=1),'ensemble':v} for k,v in ensemble.items()]).to_csv(out/'table10.csv',index=False)
    integrated_result=integrated(out,side)
    scopes=[{'scope':'Interaction, three seeds','model':'Morgan LightGBM','metric':'accuracy','value':float(t8.iloc[0].accuracy_mean),'sample_sd':float(t8.iloc[0].accuracy_sd)}, {'scope':'Interaction, historical cache','model':'Weighted GCN','metric':'accuracy','value':float(t8.iloc[1].accuracy_mean),'sample_sd':float(t8.iloc[1].accuracy_sd)}, {'scope':'Graph component analyses','model':'Separate matched protocols; not full factorial','metric':'see Tables 9 and 9a','value':None,'sample_sd':None}, {'scope':'Side-effect individual runs','model':'Weighted Morgan LightGBM','metric':'exact_match','value':individual.exact_match.mean(),'sample_sd':individual.exact_match.std(ddof=1)}, {'scope':'Side-effect ensemble','model':'Equal weights across 3 seeds','metric':'exact_match','value':ensemble['exact_match'],'sample_sd':None}, {'scope':'Integrated evaluation','model':'GCN concat 44 + side ensemble','metric':'strict accuracy, N=4390','value':integrated_result['strict_accuracy'],'sample_sd':None}]
    pd.DataFrame(scopes).to_csv(out/'table13.csv',index=False)
    status={'tables_recomputed_from_predictions':['8','10','11','12'],'table13':'derived scope summary','table9_complete':not missing,'table9_missing_settings':missing,'table9_available_rows_source':'archived per-seed metric outputs; not fresh inference','training_executed':False,'checkpoint_inference_executed':False}
    (out/'reproduction_status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status,indent=2))

if __name__=='__main__':main()
