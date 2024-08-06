import pandas as pd
import numpy as np
from employee_attrition.config import config
from employee_attrition.preprocessing.data_handling import load_dataset, save_pipeline
import employee_attrition.pipeline as pipe
from sklearn.model_selection import train_test_split
import employee_attrition.eval_metrics as em
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

def perform_training():
    data = load_dataset(config.TRAIN_FILE)
    pd.set_option('display.max_columns', None)
    train_x = data.drop(config.TARGET, axis=1)
    train_y = data[config.TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2,stratify=data['Attrition'], random_state=42)
    print('======================= Data has been loaded ==============================')

    pipe.classification_pipeline.fit(X_train, y_train)
    print('========================== model fitted =====================================')


    print('========================== model evaluated on train data =====================================')

    y_pred = pipe.classification_pipeline.predict(X_train)
    em.eval_metrics(y_train, y_pred)

    print('========================== model evaluated on test data =====================================')
    
    y_pred = pipe.classification_pipeline.predict(X_test)
    em.eval_metrics(y_test, y_pred)
    


    save_pipeline(pipe.classification_pipeline)
    print('======================== saving pipeline ===================================')

if __name__=='__main__':
    perform_training()

