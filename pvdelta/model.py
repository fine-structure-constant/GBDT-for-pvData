"""Nested grouped CV: no identifiers/outcome-derived quantities in predictors."""
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .prepare import save_table


def feature_columns(data, include_control=True):
    numeric = [c for c in data.columns if c.startswith(('A_has_','A_fraction_','B_has_','B_fraction_','X_has_','X_fraction_','additive_atoms_'))]
    numeric += [f'{s}_{k}' for s in 'ABX' for k in ['sum','ratio_known','n_species','mixing_entropy']]
    numeric += ['A_to_B','X_to_B','nominal_charge_residual_per_B','additive_mw_g_mol','additive_heavy_atom_count',
                'additive_formula_charge','additive_fragment_count','additive_cation_fragments','additive_anion_fragments',
                'additive_net_formal_charge','additive_ionic_representation','additive_smiles_missing','additive_mixture_flag']
    if include_control:
        numeric += ['pce_before']
    categorical = ['architecture','composition_topology','additive_name_canonical']
    return numeric, categorical


def make_model(numeric, categorical, params, seed):
    prep = ColumnTransformer([
        ('num',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),numeric),
        ('cat',OneHotEncoder(handle_unknown='ignore',min_frequency=5,sparse_output=True),categorical)
    ],sparse_threshold=1.0)
    return Pipeline([('preprocess',prep),('gbdt',GradientBoostingRegressor(random_state=seed,**params))])


def metrics(y,p):
    return {'MAE_pp':float(mean_absolute_error(y,p)), 'RMSE_pp':float(np.sqrt(mean_squared_error(y,p))),
            'R2':float(r2_score(y,p)), 'sign_accuracy':float(np.mean((np.asarray(y)>0)==(np.asarray(p)>0)))}


def grouped_bootstrap(y,p,groups,seed=42,repeats=1000):
    rng=np.random.default_rng(seed)
    unique=np.unique(groups)
    indices={g:np.flatnonzero(groups==g) for g in unique}
    values=[]
    for _ in range(repeats):
        idx=np.concatenate([indices[g] for g in rng.choice(unique,len(unique),replace=True)])
        values.append([mean_absolute_error(y[idx],p[idx]),np.sqrt(mean_squared_error(y[idx],p[idx])),r2_score(y[idx],p[idx])])
    arr=np.asarray(values)
    return {key:[float(x) for x in np.percentile(arr[:,i],[2.5,97.5])] for i,key in enumerate(['MAE_pp','RMSE_pp','R2'])}


def train(data, config, out):
    frame=data.loc[data.strict_eligible].copy().reset_index(drop=True)
    if len(frame)<100:
        raise ValueError('Too few strict records; review exclusions before training')
    numeric,categorical=feature_columns(frame)
    features=numeric+categorical
    forbidden=['pce_after','pce_delta_abs','pce_delta_relative_pct','delta_recomputed_pp','relative_recomputed_pct','source_excel_row','pubchem_id','cas_number','issue_codes']
    assert not set(forbidden)&set(features)
    for c in numeric:
        frame[c]=pd.to_numeric(frame[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
    x=frame[features]
    y=frame[config['target']].to_numpy(float)
    groups=frame.material_pair_group.to_numpy()
    outer=GroupKFold(n_splits=config['cv_folds'])
    results=[]
    fold_results=[]
    tuning=[]
    importance=[]
    pred={name:np.full(len(frame),np.nan) for name in ['GBDT','Dummy_mean','Dummy_median','Control_only_GBDT','Materials_only_GBDT']}
    folds=np.zeros(len(frame),dtype=int)
    feature_groups={
        'Control PCE':['pce_before'],
        'A site':[c for c in features if c.startswith('A_')],
        'B site':[c for c in features if c.startswith('B_')],
        'X site':[c for c in features if c.startswith('X_')],
        'Host topology':['composition_topology','nominal_charge_residual_per_B'],
        'Architecture':['architecture'],
        'Additive name':['additive_name_canonical'],
        'Additive chemistry':[c for c in features if c.startswith('additive_') and c!='additive_name_canonical']
    }
    for fold,(tr,te) in enumerate(outer.split(x,y,groups),1):
        assert not set(groups[tr])&set(groups[te])
        inner=GroupKFold(n_splits=3)
        scores=[]
        for ci,params in enumerate(config['gbdt_candidates']):
            vals=[]
            for itr,iva in inner.split(x.iloc[tr],y[tr],groups[tr]):
                m=make_model(numeric,categorical,params,config['seed'])
                m.fit(x.iloc[tr[itr]],y[tr[itr]])
                vals.append(mean_absolute_error(y[tr[iva]],m.predict(x.iloc[tr[iva]])))
            scores.append(float(np.mean(vals)))
            tuning.append({'outer_fold':fold,'candidate':ci,'inner_group_MAE_pp':scores[-1],'parameters_json':json.dumps(params)})
        chosen=int(np.argmin(scores))
        params=config['gbdt_candidates'][chosen]
        model=make_model(numeric,categorical,params,config['seed'])
        model.fit(x.iloc[tr],y[tr])
        pred['GBDT'][te]=model.predict(x.iloc[te])
        folds[te]=fold
        for strategy in ['mean','median']:
            dummy=DummyRegressor(strategy=strategy).fit(x.iloc[tr],y[tr])
            pred['Dummy_'+strategy][te]=dummy.predict(x.iloc[te])
        control=make_model(['pce_before'],[],params,config['seed'])
        control.fit(x.iloc[tr],y[tr])
        pred['Control_only_GBDT'][te]=control.predict(x.iloc[te])
        no_control=make_model([c for c in numeric if c!='pce_before'],categorical,params,config['seed'])
        no_control.fit(x.iloc[tr],y[tr])
        pred['Materials_only_GBDT'][te]=no_control.predict(x.iloc[te])
        for name in pred:
            fold_results.append({'validation':'material_pair_group','model':name,'fold':fold,'n_train':len(tr),'n_test':len(te),'chosen_candidate':chosen,**metrics(y[te],pred[name][te])})
        base_mae=mean_absolute_error(y[te],pred['GBDT'][te])
        rng=np.random.default_rng(config['seed']+fold)
        for label,cols in feature_groups.items():
            for rep in range(5):
                shuffled=x.iloc[te].copy()
                order=rng.permutation(len(te))
                shuffled.loc[:,cols]=shuffled[cols].iloc[order].to_numpy()
                value=mean_absolute_error(y[te],model.predict(shuffled))-base_mae
                importance.append({'feature_group':label,'fold':fold,'repeat':rep,'delta_MAE_pp':float(value)})
        print(f'Outer fold {fold}: n={len(te)}, candidate={chosen}, MAE={base_mae:.4f}',flush=True)
    for name,p in pred.items():
        results.append({'validation':'material_pair_group','model':name,'n':len(frame),**metrics(y,p)})
    for validation,splitter,vgroups in [
        ('unseen_additive_group',GroupKFold(n_splits=config['cv_folds']),frame.additive_group.to_numpy()),
        ('random_row_diagnostic',KFold(n_splits=config['cv_folds'],shuffle=True,random_state=config['seed']),None)
    ]:
        # Fixed before evaluation: these are sensitivity tests, not tunable tests.
        p=np.full(len(frame),np.nan)
        base=np.full(len(frame),np.nan)
        for fold,(tr,te) in enumerate(splitter.split(x,y,vgroups),1):
            if vgroups is not None:
                assert not set(vgroups[tr])&set(vgroups[te])
            m=make_model(numeric,categorical,config['gbdt_candidates'][0],config['seed'])
            m.fit(x.iloc[tr],y[tr])
            p[te]=m.predict(x.iloc[te])
            base[te]=np.mean(y[tr])
            fold_results.append({'validation':validation,'model':'GBDT_fixed','fold':fold,'n_train':len(tr),'n_test':len(te),'chosen_candidate':0,**metrics(y[te],p[te])})
        results.append({'validation':validation,'model':'GBDT_fixed','n':len(frame),**metrics(y,p)})
        results.append({'validation':validation,'model':'Dummy_mean','n':len(frame),**metrics(y,base)})
        pred[validation]=p
        print(f'{validation}: {metrics(y,p)}',flush=True)
    predictions=frame[['record_id','source_excel_row','material_pair_group','additive_group','pce_before','pce_after',config['target'],'architecture','composition_normalized','additive_name_canonical']].copy()
    predictions['outer_fold']=folds
    for name,p in pred.items():
        predictions['prediction_'+name]=p
    predictions['residual_pp']=y-pred['GBDT']
    predictions['abs_error_pp']=abs(predictions.residual_pp)
    result_frame=pd.DataFrame(results)
    save_table(result_frame,out/'model_metrics.csv')
    save_table(pd.DataFrame(fold_results),out/'fold_metrics.csv')
    save_table(pd.DataFrame(tuning),out/'hyperparameter_search.csv')
    save_table(predictions,out/'oof_predictions.csv')
    imp=pd.DataFrame(importance)
    save_table(imp,out/'permutation_importance_repeats.csv')
    imp_summary=imp.groupby('feature_group').delta_MAE_pp.agg(['mean','std']).sort_values('mean',ascending=False).reset_index()
    save_table(imp_summary,out/'permutation_importance.csv')
    # Final hyperparameter choice uses an independent full-data grouped CV for
    # deployment only. Published performance is strictly outer OOF above.
    final_scores=[]
    for ci,params in enumerate(config['gbdt_candidates']):
        vals=[]
        for tr,te in GroupKFold(n_splits=3).split(x,y,groups):
            m=make_model(numeric,categorical,params,config['seed']).fit(x.iloc[tr],y[tr])
            vals.append(mean_absolute_error(y[te],m.predict(x.iloc[te])))
        final_scores.append(float(np.mean(vals)))
    selected=int(np.argmin(final_scores))
    final_model=make_model(numeric,categorical,config['gbdt_candidates'][selected],config['seed']).fit(x,y)
    joblib.dump({'pipeline':final_model,'feature_columns':features,'numeric_columns':numeric,'categorical_columns':categorical,'target':config['target'],'unit':'percentage_points','config':config},out/'gbdt_model.joblib')
    save_table(frame[['record_id','source_excel_row',config['target'],'material_pair_group','additive_group']+features],out/'training_features.csv')
    specs=[]
    for c in features:
        meaning='组成名义位内比例，不是实测占位' if '_fraction_' in c else '该物种是否存在；未知不等于0' if '_has_' in c else '添加剂分子式的元素原子数' if 'additive_atoms_' in c else '已知对照效率，单位百分点' if c=='pce_before' else '名义结构/配方/组成或添加剂描述符，见方法报告'
        specs.append({'feature':c,'type':'numeric' if c in numeric else 'categorical','meaning':meaning,'missing_count_strict':int(frame[c].isna().sum()),'n_unique_strict':int(frame[c].nunique()),'used_as_predictor':True})
    save_table(pd.DataFrame(specs),out/'feature_dictionary.csv')
    summary={'n':len(frame),'n_material_pair_groups':int(frame.material_pair_group.nunique()),'n_additive_groups':int(frame.additive_group.nunique()),'n_raw_features':len(features),
             'target':config['target'],'unit':'percentage points, not relative percent','primary_validation':'5-fold outer material-pair GroupKFold, inner 3-fold candidate selection',
             'primary_metrics':metrics(y,pred['GBDT']),'primary_group_bootstrap_95pct':grouped_bootstrap(y,pred['GBDT'],groups,config['seed']),
             'final_parameters':config['gbdt_candidates'][selected],'final_candidate_group_cv_mae':final_scores,
             'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__,'pandas':pd.__version__,'numpy':np.__version__},
             'limitations':['No DOI: material-pair groups cannot guarantee paper independence','No dose/location/process: association model, not causal additive effect','Explicit SMILES fragment extraction is lexical, no chemical-graph validation','Bootstrap intervals condition on OOF predictions; not full nested-resampling uncertainty']}
    (out/'model_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(result_frame.to_string(index=False),flush=True)
    return result_frame,predictions,imp_summary,summary
