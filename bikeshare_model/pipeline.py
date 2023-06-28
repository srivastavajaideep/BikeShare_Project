import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import *

'''
Steps in the bikeshare_pipeline:
1. Impute the weathersit column to fill null values with most frequent value
2. Map the ordinal categorical columns [yr, mnth, season, weathersit, holday, workingday, hr]
3. Remove outliers from the numerical columns [temp, atemp, hum, windspeed]
4. Apply the StandardScalar 
5. Fit the Random Forest Regression model to predict the target variable 
'''


bikeshare_pipeline = Pipeline([
                     # === Imputation ===
                     ("1. weathersit_imputation", WeathersitImputer(variables=config.model_config.weathersit_var)),
                     # === Mapper ===
                     ("2.1. year_mapping",Mapper(config.model_config.year_var, config.model_config.year_mappings)),
                     ("2.2. month_mapping",Mapper(config.model_config.month_var, config.model_config.month_mappings)),
                     ("2.3. season_mapping",Mapper(config.model_config.season_var, config.model_config.season_mappings)),
                     ("2.4. weather_mapping",Mapper(config.model_config.weathersit_var, config.model_config.weather_mappings)),
                     ("2.5. holiday_mapping",Mapper(config.model_config.holiday_var, config.model_config.holiday_mappings)),
                     ("2.6. workingday_mapping",Mapper(config.model_config.workingday_var, config.model_config.workingday_mappings)),
                     ("2.7. hour_mapping",Mapper(config.model_config.hour_var, config.model_config.hour_mappings)),
                     # === Outlier Removal ===
                     ("3.1. temp_outlier", OutlierHandler(variables=config.model_config.temp_var)),
                     ("3.2. atemp_outlier", OutlierHandler(variables=config.model_config.atemp_var)),
                     ("3.3. humidity_outlier", OutlierHandler(variables=config.model_config.humidity_var)),
                     ("3.4. windspeed_outlier", OutlierHandler(variables=config.model_config.windspeed_var)),
                     # === Standard Scaling ===
                     ("4. standard_scaler", StandardScaler()),
                     # === Random Forest Regression Model ===
                     ('5. rf_model', RandomForestRegressor(n_estimators=config.model_config.n_estimators, max_depth=config.model_config.max_depth,
                                      random_state=config.model_config.random_state))
])
