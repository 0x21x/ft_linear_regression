import json
import pandas as pd
from typing import Tuple, Optional

def get_model_parameters(intercept: Optional[float] = None, slope: Optional[float] = None,
    km_min: Optional[float] = None, km_max: Optional[float] = None, price_min: Optional[float] = None,
    price_max: Optional[float] = None) -> Tuple[float, float, float, float, float, float]:
    try:
        with open('model_parameters.json', 'r') as file:
            data: dict = json.load(file)
            _intercept: float = data['intercept'] or 0
            _slope: float = data['slope'] or 0
            _km_min: float = data['km_min'] or 0
            _km_max: float = data['km_max'] or 0
            _price_min: float = data['price_min'] or 0
            _price_max: float = data['price_max'] or 0
    except (json.decoder.JSONDecodeError, KeyError, FileNotFoundError):
        print('The model has not been trained yet.')
        _intercept: float = intercept or 0
        _slope: float = slope or 0
        _km_min: float = km_min or 0
        _km_max: float = km_max or 0
        _price_min: float = price_min or 0
        _price_max: float = price_max or 0
        with open('model_parameters.json', 'w') as file:
            json.dump({'intercept': intercept, 'slope': slope, 'km_min': km_min, 'km_max': km_max, 'price_min': price_min, 'price_max': price_max}, file, indent=4)
    return _intercept, _slope, _km_min, _km_max, _price_min, _price_max

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

def get_dataset_parameters() -> Tuple[float, float, float, float]:
    with open('model_parameters.json', 'r') as file:
        data: dict = json.load(file)
        km_min: float = data['km_min'] or 0
        km_max: float = data['km_max'] or 0
        price_min: float = data['price_min'] or 0
        price_max: float = data['price_max'] or 0
    return km_min, km_max, price_min, price_max

def save_dataset_parameters(km_min: float, km_max: float, price_min: float, price_max: float) -> None:
    with open('model_parameters.json', 'r') as file:
        data: dict = json.load(file)
        data['km_min'] = km_min
        data['km_max'] = km_max
        data['price_min'] = price_min
        data['price_max'] = price_max
    with open('model_parameters.json', 'w') as file:
        json.dump(data, file, indent=4)

def parse_and_check_csv(normalized:bool=True) -> pd.DataFrame:
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

    km_min, km_max, price_min, price_max = float(data['km'].min()), float(data['km'].max()), float(data['price'].min()), float(data['price'].max())
    get_model_parameters(None, None, km_min=km_min, km_max=km_max, price_min=price_min, price_max=price_max)
    if normalized:
        data['km'] = (data['km'] - km_min) / (km_max - km_min)
        data['price'] = (data['price'] - price_min) / (price_max - price_min)
    return data

def save_model_parameters(intercept: float, slope: float) -> None:
    with open('model_parameters.json', 'r') as file:
        data: dict = json.load(file)
        km_min = data['km_min'] or 0
        km_max = data['km_max'] or 0
        price_min = data['price_min'] or 0
        price_max = data['price_max'] or 0
    with open('model_parameters.json', 'w') as file:
        json.dump({'intercept': intercept, 'slope': slope, 'km_min': km_min, 'km_max': km_max, 'price_min': price_min, 'price_max': price_max}, file, indent=4)
