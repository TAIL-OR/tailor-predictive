from flask import Flask,request, jsonify 
import pandas as pd
import subprocess
import os
from utils import months

# get the parent directory of the current path
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)

models_path = os.path.join(parent_directory, 'models')
scripts_path = os.path.join(parent_directory, 'scripts')
results_path = os.path.join(parent_directory, 'results')

app = Flask(__name__)
app.config["DEBUG"] = True

data_predicted = pd.read_csv(os.path.join(results_path, 'forecast_full_lgbm.csv'))
max_date = data_predicted['ds'].max()

print(int(max_date.split('-')[1]))

@app.route('/predict', methods=['GET'])
def predict(): 
    """
    This function will return the forecast for the month passed as a parameter
    month: int - The month to forecast
    """
    month = request.args.get('month')
    if month is None:
        return 400
    
    data_predicted = pd.read_csv('../results/forrecast_full_lgbm.csv')
    max_date = data_predicted['ds'].max()

    if month > max_date.split('-')[1]:
        subprocess.run(['python3', os.path.join(scripts_path, 'update_data.py')])
        subprocess.run(['python3', os.path.join(models_path, 'run_model.py'), '--model', 'lgbm', '--n_trials', '10', '--horizon', '30'])
        
        NEED_UPDATE = False
        data_predicted = pd.read_csv(os.path.join(results_path, f'forrecast_full_lgbm_{month}.csv'))  # Use the 'month' parameter to construct the file path

    return jsonify(data_predicted.to_dict())

    