from typing import List
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

class DebugTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, step_name: str):
        self.step_name = step_name

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        print('--------------------------------------------------------------------')
        print(' ****** Debugging step *********')
        print(X)
        print('--------------------------------------------------------------------')
    
        return X
class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables: list[str]):
        if not isinstance(variables, list):
            raise ValueError(' variables should be a list')
        
        self.variables = variables
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self
    
    def transform(self, X: pd.DataFrame)-> pd.DataFrame:
        X = X.copy()

        dummies = pd.get_dummies(X[self.variables], dtype=int)
        X = X.drop(self.variables, axis=1).join(dummies)

        return X



class LogTransform(BaseEstimator, TransformerMixin):
    def __init__(self, variables: list[str]):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        self.variables = variables
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):

        return self
    
    def transform(self, X: pd.DataFrame)  -> pd.DataFrame :

        X = X.copy()

        k = 1e-6
        for var in self.variables:
            X[var] = np.log(X[var] + k)
        return X
    
    
class Drop(BaseEstimator, TransformerMixin):
    """ Delete the features """
    def __init__(self, variables: list[str]):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        X = X.drop(self.variables, axis=1)
      
        return X


class Mapper(BaseEstimator, TransformerMixin):
    """ Categorical variable mapper """

    def __init__(self, variables: List[str], mappings: dict):

        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        
        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame :
        X = X.copy()

        for feature in self.variables:
            print()
            X[feature] = X[feature].map(self.mappings[feature])
        
        return X


from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class Over_sampling(BaseEstimator, TransformerMixin):
    def __init__(self, step_name: str):
        self.step_name = step_name
        self.smote = SMOTE()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X_resampled_, self.y_resampled_ = self.smote.fit_resample(X, y)
        return self

    def transform(self, X: pd.DataFrame):
        # Use resampled data
        if not hasattr(self, 'X_resampled_'):
            raise ValueError("The fit method must be called before transform.")
        return self.X_resampled_
