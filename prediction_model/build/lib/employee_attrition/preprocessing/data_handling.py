import os
import joblib
import pandas as pd
from employee_attrition.config import config

## load the dataset

def load_dataset(filename):
    filepath = os.path.join(config.DATAPATH, filename)
    _data = pd.read_csv(filepath)
    print(f' {filename} has benn loaded')
    return _data

## serialization
def save_pipeline(pipeline_to_save):
    save_path = os.path.join(config.MODEL_PATH, config.MODEL_NAME)
    joblib.dump(pipeline_to_save, save_path)
    print('*'* 50)
    print(f' Model has been saved under the name {config.MODEL_NAME}')
    print('*'* 50)

## deserialization 

def load_pipeline(pipeline_to_load):
    save_path = os.path.join(config.MODEL_PATH, config.MODEL_NAME)
    model_loaded = joblib.load(save_path)
    print(' Model has been loaded')
    return model_loaded


