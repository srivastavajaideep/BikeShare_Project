
"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
import math
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from bikeshare_model.config.core import config
from bikeshare_model.processing.features import OutlierHandler, WeathersitImputer


def test_weathersit_imputer_transformer(sample_input_data):
    # Given
    transformer = WeathersitImputer(
        variables=config.model_config.weekday_var,
    )
    assert np.isnan(sample_input_data.loc[5,'weekday'])

    # When
    subject = transformer.fit(sample_input_data).transform(sample_input_data)

    # Then
    assert subject.loc[5,'weekday'] == "Sun"
    
    
    
def test_temp_outlier_handler(sample_input_data):
    # Given
    transformer = OutlierHandler(
        variables=config.model_config.temp_var,
    )
    assert math.isnan(sample_input_data.loc[5,'temp'])== False

    # When
    subject = transformer.fit(sample_input_data).transform(sample_input_data)

    # Then
    assert subject.loc[5,'temp'] <= 43
    assert subject.loc[5,'temp'] >= -13 
   