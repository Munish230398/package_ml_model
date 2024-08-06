import joblib
import pandas as pd
import numpy as np

from employee_attrition.preprocessing.data_handling import load_dataset,load_pipeline
from employee_attrition.config import config

classification_pipeline = load_pipeline(config.MODEL_NAME)

def generate_predictions():
    test_data = load_dataset(config.TEST_FILE)
    pred = classification_pipeline.predict(test_data)
    return {'predictions': pred}


if __name__=='__main__':
    generate_predictions()

