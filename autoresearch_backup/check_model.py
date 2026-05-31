import pickle
import numpy as np

for name in ['geock_deep_trees_final.pkl', 'geock_deep_trees_no2016.pkl', 'geock_v2_xgboost_39k.pkl']:
    with open(name,'rb') as f:
        d = pickle.load(f)
    cfg = d.get('config', {})
    model = d.get('model', d.get('xgb'))
    print(f'=== {name} ===')
    print(f'  CV R = {d.get("cv_r",0):.4f}')
    print(f'  n_features = {d.get("n_features","?")}')
    print(f'  n_samples = {d.get("n_samples","?")}')
    print(f'  selector = {type(d.get("selector",None)).__name__}')
    scaler = d.get('scaler',None)
    if scaler:
        print(f'  scaler is_fitted = {hasattr(scaler, "mean_")}')
    if hasattr(model, 'get_params'):
        params = model.get_params()
        for k in ['max_depth','n_estimators','learning_rate','reg_alpha','reg_lambda']:
            print(f'  {k} = {params.get(k)}')
    print()
