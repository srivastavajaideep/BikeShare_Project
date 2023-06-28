
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from bikeshare_model import __version__ as _version
from bikeshare_model.config.core import config
from bikeshare_model.pipeline import bikeshare_pipeline
from bikeshare_model.processing.data_manager import load_pipeline, pre_pipeline_preparation
from bikeshare_model.processing.validation import validate_inputs

# load the saved pipeline
pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
bikeshare_pipeline= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data: dict) -> dict:
    """Make a prediction using a saved model """
    
    validated_data,errors= validate_inputs(input_df=pd.DataFrame(input_data))
  
    #data = pre_pipeline_preparation(data_frame=pd.DataFrame(input_data))
    #print(data.info())
    validated_data=validated_data.reindex(columns=['season','hr','holiday','workingday','weathersit','temp','atemp','hum','windspeed',
                               'yr','mnth','weekday_Fri','weekday_Mon','weekday_Sat','weekday_Sun',
                               'weekday_Thu','weekday_Tue','weekday_Wed'])      
    
    results = {"predictions": None, "version": _version, "errors" : errors}
    
    # predictions = bikeshare_pipeline.predict(data)

    # results = {"predictions": predictions,"version": _version}
    # print(results)
    if not errors:
     print("Test method")
     predictions=bikeshare_pipeline.predict(validated_data)
     results={"predictions":predictions,"version":_version, "errors": errors}
    print("Condition check",results)
    return results

# if __name__ == "__main__":
    
#     data_in={'dteday': ['2011-06-03'],
#              'season': ['summer'],
#              'hr': ['2am'],
#              'holiday': ['No'],
#              'weekday': ['Fri'],
#              'workingday': ['Yes'],
#              'weathersit': ['Clear'],
#              'temp': [18.320000000000004],
#              'atemp': [18.9998],
#              'hum': [43.0],
#              'windspeed': [15.001299999999999],
#              'casual': [0],
#              'registered': [12]
#              }

#     make_prediction(input_data=data_in)
    