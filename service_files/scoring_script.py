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
    model_path = Model.get_model_path('ClusterAlgorithm_01')
    ref_cols, predictor = joblib.load(model_path)

# run function for predictions
def run(raw_data):
    # load data
    data_dict = json.loads(raw_data)

    # convert to dataframe
    #data = pd.DataFrame.from_dict(data_dict)
    data = pd.DataFrame.from_dict(data_dict)

    # hot encode once again
    '''
    data_enc = pd.get_dummies(data)

    # columns reference
    deploy_cols = data_enc.columns

    # compare to get number of missing columns
    missing_cols = ref_cols.difference(deploy_cols)

    # check for missing colums
    for cols in missing_cols:
        data_enc[cols] = 0

    # insert all columns
    data_enc = data_enc[ref_cols]
    '''
    # prediction
    #predictions = predictor.predict(data_enc)
    algo_model = predictor(sklearn_load_ds=data)
    predictions = algo_model.Kmeans(output='replace').classify()

  

    '''
    
    # returned result format
    classes = ['Abnormal', 'Normal']
    # list of predictions
    predicted_classes = []
    
    for prediction in predictions:
        predicted_classes.append(classes[prediction])
    '''

    return json.dumps(predictions)