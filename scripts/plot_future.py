import pandas as pd
import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib.pyplot as plt
import numpy as np

# Load the data
past_data = pd.read_csv('../data/dados-gerais.csv')
future_data = pd.read_csv('../results/forecast_full_xgboost.csv')
future_data = future_data.rename(columns={'XGBRegressor': 'case_cnt', 'unique_id': 'ra', 'ds': 'date'})
future_data_lgbm = pd.read_csv('../results/forecast_full_lgbm.csv')
future_data_lgbm = future_data_lgbm.rename(columns={'LGBMRegressor': 'case_cnt', 'unique_id': 'ra', 'ds': 'date'})
future_data_delphi = pd.read_csv('../results/forecast_delphi.csv')
future_data_delphi = future_data_delphi.rename(columns={'Total Detected': 'case_cnt', 'Province': 'ra', 'Day': 'date'})
future_data_delphi = future_data_delphi[['ra', 'date', 'case_cnt']]

past_data['date'] = pd.to_datetime(past_data['date'])
future_data['date'] = pd.to_datetime(future_data['date'])
future_data_lgbm['date'] = pd.to_datetime(future_data_lgbm['date'])
future_data_delphi['date'] = pd.to_datetime(future_data_delphi['date'])

past_data_last_4_months = past_data[past_data['date'] >= past_data['date'].max() - pd.DateOffset(months=4)]

# Plot the data
output_path = '../models/plots/delphi/'
for ra in past_data_last_4_months['ra'].unique():
    plt.plot(past_data_last_4_months[past_data_last_4_months['ra'] == ra]['date'], past_data_last_4_months[past_data_last_4_months['ra'] == ra]['case_cnt'], color='blue', label='Past Data')
    #plt.plot(future_data[future_data['ra'] == ra]['date'], future_data[future_data['ra'] == ra]['case_cnt'], color='red', label='Future Data')
    #plt.plot(future_data_lgbm[future_data_lgbm['ra'] == ra]['date'], future_data_lgbm[future_data_lgbm['ra'] == ra]['case_cnt'], color='purple', label='Future Data LGBM')
    plt.plot(future_data_delphi[future_data_delphi['ra'] == ra]['date'], future_data_delphi[future_data_delphi['ra'] == ra]['case_cnt'], color='green', label='Future Data Delphi')
    plt.xlabel('Time')
    plt.ylabel('Case Count')
    plt.title(f'RA {ra} - Past (last 4 months) and Future Case Count')
    plt.legend()
    plt.savefig(f'{output_path}/plot_{ra}.png')
    plt.clf()