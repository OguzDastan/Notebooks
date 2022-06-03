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
    model_path = Model.get_model_path('ISOF02')
    ref_cols, predictor = joblib.load(model_path)

# run function for predictions


def run(raw_data):

    # load data
    data_dict = json.loads(raw_data)

    # convert to dataframe
    data = pd.DataFrame.from_dict(data_dict)


    data['scores'] = predictor.decision_function(data[['FanSpeed', 
                            'Temp_Room', 'Temp_Out', 'Temp_Floor']])

    data['anomaly'] = predictor.predict(data[['FanSpeed', 
                            'Temp_Room', 'Temp_Out', 'Temp_Floor']])

    scores = data.loc[((data['scores'] < 0) 
                        & (data['FanSpeed'] > 0) 
                        & (data['ControllerStateNumber'] > 0) 
                        & (data['Temp_Out'] < 26)
                        & (data['ControllerStateNumber'] != 71.0) 
                        & (data['ControllerStateNumber'] != 71.1)
                        & (data['ControllerStateNumber'] != 72.0)
                        & (data['ControllerStateNumber'] != 73.0)
                        & (data['ControllerStateNumber'] != 74.0)
                        & (data['ControllerStateNumber'] != 74.4)
                        & (data['ControllerStateNumber'] != 71.4)
                        )]

    result = (scores['anomaly'].count() / data['anomaly'].count()) * 100

    return json.dumps(result)