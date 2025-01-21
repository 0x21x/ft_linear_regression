import pandas as pd
from matplotlib import pyplot as plt
from utils import get_model_parameters, parse_and_check_csv

def plot(data: pd.DataFrame, intercept: float, slope: float) -> None:
    plt.scatter(data['km'], data['price'])
    plt.plot(data['km'], intercept + slope * data['km'], color='red')
    plt.show()

if __name__ == '__main__':
    intercept, slope, km_min, km_max, price_min, price_max = get_model_parameters()
    data = parse_and_check_csv(normalized=False)
    plot(data, intercept, slope)
