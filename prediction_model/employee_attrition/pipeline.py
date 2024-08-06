from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


from employee_attrition.config import config
import employee_attrition.preprocessing.preprocessing as pp
from sklearn.linear_model import LogisticRegression


classification_pipeline = Pipeline(
    [
        ## ===== Drop irrelevent features =====
        ('Drop',pp.Drop(variables=config.FEATURES_TO_DROP)),

        ## ===== Map categorical columns =======
        ('Mapper',pp.Mapper(variables=config.FEATURES_TO_MAP, mappings = config.MAPPINGS)),
        

        ## ===== encode the categorical columns =======
        ('Encoder',pp.Encoder(variables=config.FEATURES_TO_ENCODE)),


        ## ===== log transform the columns =======
        ('transformer',pp.LogTransform(variables=config.FEATURES_TO_LOG_TRANSFORM)),

        ## =========  fitting the model ================
        ('model', LogisticRegression(class_weight='balanced', max_iter=3000)),


    ]
)