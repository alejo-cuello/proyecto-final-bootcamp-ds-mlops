import json
import os
import mlflow.sklearn
import numpy as np
import pandas as pd

from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

data_path = 'data/split_by_l2'
experiment_name = 'baseline-model-rfr'
file_name = 'l2_capital_federal.csv'

runs = mlflow.search_runs(
    experiment_names=[experiment_name],
    filter_string=f"tags.mlflow.runName = 'rfr_{file_name}'"
)

runs['rmse_delta'] = runs['metrics.rmse_test'] - runs['metrics.rmse_train']
best_run = runs.loc[[runs["rmse_delta"].idxmax()]]

data_encoded = pd.read_csv(os.path.join(data_path, file_name), sep=',', index_col=0)

with open("notebooks/price_by_quantile.json", "rb") as handle:
    price_by_quantile = json.load(handle)

with open("scripts/output/best_random_params_"+file_name.removesuffix(".csv")+".txt", "w") as f:
    f.write(f'l2: {file_name}\n')
    f.write(f'run_id: {best_run[['run_id']].values[0]}\n')
    f.write(f'param_bootstrap:{best_run[['params.param_bootstrap']].values[0]}\n')
    f.write(f'param_max_depth:{best_run[['params.param_max_depth']].values[0]}\n')
    f.write(f'param_max_features:{best_run[['params.param_max_features']].values[0]}\n')
    f.write(f'param_min_samples_leaf:{best_run[['params.param_min_samples_leaf']].values[0]}\n')
    f.write(f'param_min_samples_split:{best_run[['params.param_min_samples_split']].values[0]}\n')
    f.write(f'param_n_estimators:{best_run[['params.param_n_estimators']].values[0]}\n')