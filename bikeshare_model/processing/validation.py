from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from bikeshare_model.config.core import config
from bikeshare_model.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    pre_processed = pre_pipeline_preparation(data_frame=input_df)
    validated_data = pre_processed[config.model_config.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


# class DataInputSchema(BaseModel):
#     PassengerId:Optional[int]
#     Pclass: Optional[int]
#     Name: Optional[str]
#     Sex: Optional[str]
#     Age: Optional[float]
#     SibSp: Optional[int]
#     Parch: Optional[int]
#     Ticket: Optional[str]
#     Cabin: Optional[Union[str, float]]
#     Embarked: Optional[str]
#     Fare: Optional[Union[int, float]]


class DataInputSchema(BaseModel):
    dteday:Optional[object]
    season: Optional[object]
    hr: Optional[object]
    holiday: Optional[object]
    weekday: Optional[object]
    workingday: Optional[object]
    weathersit: Optional[object]
    temp: Optional[float]
    atemp: Optional[float]
    hum: Optional[float]
    windspeed: Optional[float]
    casual: Optional[int]
    registered: Optional[int]
    cnt: Optional[int]
    

class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]