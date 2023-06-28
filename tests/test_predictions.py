"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from bikeshare_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    print(sample_input_data.head())
    expected_no_predictions = 3476

    # When
    result = make_prediction(input_data=sample_input_data)

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, np.ndarray)
    #assert isinstance(predictions[0], np.int64)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    _predictions = list(predictions)
    y_true = list(sample_input_data["cnt"])    
    rms = mean_squared_error(y_true,_predictions, squared=False)
    print("RMS Value",rms)
    #accuracy = accuracy_score(_predictions, y_true)
    assert rms != 0
   
    

