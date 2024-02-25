from mlforecast import MLForecast
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from window_ops.rolling import rolling_mean, rolling_max, rolling_min
import optuna
import matplotlib.pyplot as plt
import holidays
import argparse
import logging
from sklearn.metrics import mean_absolute_error

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', '-m', type=str, required=True,
    choices=['xgboost', 'lgbm'],
    help='Model to be used. Options: xgb, lgbm'
)
parser.add_argument(
    '--n_trials', '-nt', type=int, required=True,
    help='Number of trials for the optimization'
)
parser.add_argument(
    "--horizon", '-hz', type=int, required=False, default=30,
    help="Number of days to forecast"
)

args = parser.parse_args()

MODEL = "XGBRegressor" if args.model == 'xgboost' else "LGBMRegressor"
N_TRIALS = args.n_trials
TRUE_HORIZON = args.horizon
SAVE_PLOTS = False

def wmape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()

def create_instance(models, l):
    return MLForecast(
        models=models,
        freq='D',
        lags=[1,7,l],
        date_features=['dayofweek', 'month'],
        lag_transforms={
            1: [(rolling_mean, 7), (rolling_max, 7), (rolling_min, 7)],
        },
        num_threads=6
    )

def models_(params):
    if MODEL == "LGBMRegressor":
        return [
            LGBMRegressor(
                n_estimators=500,
                learning_rate=params['learning_rate'],
                num_leaves=params['num_leaves'],
                min_data_in_leaf=params['min_data_in_leaf'],
                bagging_fraction=params['bagging_fraction'],
                colsample_bytree=params['colsample_bytree'],
                bagging_freq=1,
                random_state=42,
                verbose=-1)
        ]
    elif MODEL == "XGBRegressor":
        return [
            XGBRegressor(
                n_estimators=500,
                learning_rate=params['learning_rate'],
                max_depth=params['max_depth'],
                min_child_weight=params['min_child_weight'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                random_state=42
            )
        ]

br_holidays = holidays.country_holidays('BR')
data = pd.read_csv('../data/dados-gerais.csv')
data = data.rename(columns= {'date': 'ds', 'case_cnt': 'y', 'ra': 'unique_id'})
data['ds'] = pd.to_datetime(data['ds'])

data = data.sort_values(by=['unique_id', 'ds']).groupby('unique_id').apply(lambda x: x.fillna(method='ffill'))
data['is_holiday'] = data['ds'].dt.date.apply(lambda x: 1 if x in br_holidays else 0)

static_features = ['is_holiday']

max_date = data['ds'].max()
two_months_before = max_date - pd.DateOffset(months=2)

train = data.loc[data['ds'] < two_months_before]
valid = data.loc[data['ds'] >= two_months_before]

h = valid['ds'].nunique()
print(f'Horizon: {h}')

def objective(trial):
    
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 1e-1)
    lags = trial.suggest_int('lags', 14, 56, step=7)# step means we only try multiples of 7 starting from 14
    colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.1, 1.0)

    if MODEL == "LGBMRegressor":
        num_leaves = trial.suggest_int('num_leaves', 2, 256)
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 1, 100)
        bagging_fraction = trial.suggest_uniform('bagging_fraction', 0.1,1.0)

        models = models_({
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'min_data_in_leaf': min_data_in_leaf,
            'bagging_fraction': bagging_fraction,
            'colsample_bytree': colsample_bytree
        })
        
    elif MODEL == "XGBRegressor":
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
        subsample = trial.suggest_uniform('subsample', 0.1, 1.0)
        max_depth = trial.suggest_int('max_depth', 3, 10)

        models = models_({
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree
        })

    model = create_instance(models, lags)
    
    model.fit(train, id_col='unique_id', target_col='y', time_col='ds', static_features=static_features)
    forecast = model.predict(h=h)
    forecast = forecast.merge(valid[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'], how='left')
    forecast['y'] = forecast['y'].fillna(0)

    error = mean_absolute_error(forecast["y"], forecast[MODEL])

    return error

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=N_TRIALS)

best_params = study.best_params
print(f'Best params: {best_params}')

models = models_(best_params)
model = create_instance(models, best_params['lags'])

model.fit(train, id_col='unique_id', target_col='y', time_col='ds', static_features=static_features)

forecast = model.predict(h=h)
forecast = forecast.merge(valid[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'], how='left')
forecast['y'] = forecast['y'].fillna(0)

print(f'MAE: {mean_absolute_error(forecast["y"], forecast[MODEL])}')
print(f'WMAPE: {wmape(forecast["y"], forecast[MODEL])}')

forecast.to_csv(f'../results/forecast_{args.model}.csv', index=False)

if SAVE_PLOTS:
    plot_path = './plots/xgb/'
    for ra in forecast['unique_id'].unique():
        ra_data = forecast.loc[forecast['unique_id'] == ra]
        plt.figure(figsize=(12, 6))
        plt.plot(ra_data['ds'], ra_data['y'], label='True')
        plt.plot(ra_data['ds'], ra_data[MODEL], label='Forecast')

        plt.title(f'RA: {ra}')
        plt.legend()
        plt.savefig(f'{plot_path}{ra}.png')

model = create_instance(models, best_params['lags'])
model.fit(data, id_col='unique_id', target_col='y', time_col='ds', static_features=static_features)

forecast = model.predict(h=TRUE_HORIZON)
forecast[MODEL] = forecast[MODEL].astype(int)

forecast.to_csv(f'../results/forecast_full_{args.model}.csv', index=False)