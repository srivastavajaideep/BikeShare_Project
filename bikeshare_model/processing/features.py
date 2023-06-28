
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class WeathersitImputer(BaseEstimator, TransformerMixin):

    """WeatherSituation column Imputer"""

    def __init__(self, variables: str):

        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        self.fill_value=X[self.variables].mode()[0]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variables]=X[self.variables].fillna(self.fill_value)

        return X


class Mapper(BaseEstimator, TransformerMixin):
    """Categorical variable mapper."""

    def __init__(self, variables: str, mappings: dict):

        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variables] = X[self.variables].map(self.mappings).astype(int)

        return X
    

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Handle outliers for a single column: Instead of removing the outliers, change their values
        1. to upper-bound, if the value is higher than upper-bound, or
        2. to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, variables: str):
        
        if not isinstance(variables, str):
            raise ValueError('variables should be a str')
        
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series=None):
        # we need the fit statement to accomodate the sklearn pipeline
        q1 = X.describe()[self.variables].loc['25%']
        q3 = X.describe()[self.variables].loc['75%']
        iqr = q3 - q1
       # print("IQR",iqr)
        self.lower_bound = q1 - (1.5 * iqr)
       # print(self.lower_bound)
        self.upper_bound = q3 + (1.5 * iqr)
        #print(self.upper_bound)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for i in X.index:
            X.loc[self.variables]: self.upper_bound if X.loc[self.variables] >= self.upper_bound else self.lower_bound

        return X
