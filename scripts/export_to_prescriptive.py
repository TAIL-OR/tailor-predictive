import pandas as pd
import numpy as np

data = pd.read_csv('../results/forecat_full_xgboost.csv')
data['date'] = pd.to_datetime(data['date'])
aggregated_data = data.groupby('ra')['XGBRegressor'].sum()

