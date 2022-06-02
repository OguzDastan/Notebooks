# service script for deployment of the algorithm

#imports
import joblib
from azureml.core import Model
import json
import pandas as pd

#import pandas as pd

# init function for service deployment


def init():
    global ref_cols, predictor
    model_path = Model.get_model_path('CLA01')
    ref_cols, predictor = joblib.load(model_path)

# run function for predictions


def run(raw_data):
    # load data
    data_dict = json.loads(raw_data)
    # convert to dataframe
    #data = pd.DataFrame.from_dict(data_dict)
    data = pd.DataFrame.from_dict(data_dict)
    algo_model = predictor(sklearn_load_ds = data)
    predictions = algo_model.Kmeans(output = 'replace').classify()
    return json.dumps(predictions)