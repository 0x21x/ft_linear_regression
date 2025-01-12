import pandas as pd
from matplotlib import pyplot as plt
from typing import Tuple
from utils import get_model_parameters, get_hyperparameters, parse_and_check_csv

def plot(data: pd.DataFrame, intercept: float, slope: float) -> None:
    plt.scatter(data['km'], data['price'])
    plt.plot(data['km'], intercept + slope * data['km'], color='red')
    plt.show()

if __name__ == '__main__':
    intercept, slope = get_model_parameters()
    data: pd.DataFrame = parse_and_check_csv()
    plot(data, intercept, slope)
