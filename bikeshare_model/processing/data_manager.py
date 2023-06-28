
import sys
import typing as t
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from bikeshare_model import __version__ as _version
from bikeshare_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def get_year_and_month(dataframe: pd.DataFrame) -> pd.DataFrame:

    df = dataframe.copy()
    # convert 'dteday' column to Datetime datatype
    df['dteday'] = pd.to_datetime(df['dteday'], format='%Y-%m-%d')
    # Add new features 'yr' and 'mnth
    df['yr'] = df['dteday'].dt.year
    df['mnth'] = df['dteday'].dt.month_name()
    
    return df


def weekday_imputer(dataframe: pd.DataFrame) -> pd.DataFrame:

    df = dataframe.copy()
    wkday_null_idx = df[df['weekday'].isnull() == True].index
    # get the day name from the date
    df.loc[wkday_null_idx, 'weekday'] = df.loc[wkday_null_idx, 'dteday'].dt.day_name().apply(lambda x: x[:3])

    return df


def weekday_onehot_encoder(dataframe: pd.DataFrame) -> pd.DataFrame:

    df = dataframe.copy()

    # applying One-Hot encoder
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(df[['weekday']])

    encoded_weekday = encoder.transform(df[['weekday']])
    # get encoded feature names
    enc_wkday_features = encoder.get_feature_names_out(['weekday'])
    # append encoded weekday features to df
    df[enc_wkday_features] = encoded_weekday

    return df


def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:

    # create year and month columns
    data_frame = get_year_and_month(dataframe=data_frame)

    # impute and then one-hot encode the weekday column
    data_frame = weekday_imputer(dataframe=data_frame)
    data_frame = weekday_onehot_encoder(dataframe=data_frame)

    # drop unnecessary variables
    data_frame.drop(labels=config.model_config.unused_fields, axis=1, inplace=True)

    return data_frame


def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe


def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_preparation(data_frame=dataframe)

    return transformed


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
