import json
import pandas as pd
from typing import Tuple

def get_model_parameters() -> Tuple[float, float]:
    try:
        with open('model_parameters.json', 'r') as file:
            data: dict = json.load(file)
            intercept: float = data['intercept'] or 0
            slope: float = data['slope'] or 0
    except (json.decoder.JSONDecodeError, KeyError, FileNotFoundError):
        print('The model has not been trained yet.')
        intercept: float = 0
        slope: float = 0
        with open('model_parameters.json', 'w') as file:
            json.dump({'intercept': intercept, 'slope': slope}, file, indent=4)
    return intercept, slope

def get_hyperparameters() -> Tuple[float, int]:
    try:
        with open('hyper_parameters.json', 'r') as file:
            data: dict = json.load(file)
            learning_rate: float = data['learning_rate'] or 0.01
            epochs: int = data['epochs'] or 100
    except (json.decoder.JSONDecodeError, KeyError):
        print('The hyperparameters have not been set yet.')
        learning_rate: float = 0.01
        epochs: int = 100
        with open('hyper_parameters.json', 'w') as file:
            json.dump({'learning_rate': learning_rate, 'epochs': epochs}, file, indent=4)
    except FileNotFoundError:
        print('Parameters file not found.')
        raise SystemExit
    return learning_rate, epochs

def parse_and_check_csv() -> pd.DataFrame:
    try:
        data: pd.DataFrame = pd.read_csv('data.csv')
        if data.columns.size != 2:
            print('The CSV file must have exactly 2 columns.')
            raise SystemExit
        if data.columns[0] != 'km' or data.columns[1] != 'price':
            print('The columns must be named "km" and "price".')
            raise SystemExit
    except FileNotFoundError:
        print('CSV file not found.')
        raise SystemExit
    except pd.errors.EmptyDataError:
        print('The CSV file is empty.')
        raise SystemExit
    except pd.errors.ParserError:
        print('The CSV file is not properly formatted.')
        raise SystemExit

    for column in data.columns:
        if data[column].isnull().values.any():
            print('The CSV file must not contain any NaN values.')
            raise SystemExit
        if not data[column].dtype == 'float64' and not data[column].dtype == 'int64':
            print('The columns must contain only numeric values.')
            raise SystemExit
    return data

def save_model_parameters(intercept: float, slope: float) -> None:
    with open('model_parameters.json', 'w') as file:
        json.dump({'intercept': intercept, 'slope': slope}, file, indent=4)
