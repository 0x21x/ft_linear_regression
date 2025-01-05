import pandas as pd
from typing import Tuple, List
from math import pow
from utils import get_model_parameters, get_hyperparameters, parse_and_check_csv, save_model_parameters

def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    return data.assign(
        km=(data['km'].astype(float) - data['km'].mean()) / data['km'].std(),
        price=(data['price'].astype(float) - data['price'].mean()) / data['price'].std()
    )

def predict(mileage: float, intercept: float, slope: float) -> float:
    return intercept + (mileage * slope)

def cost_function(data: pd.DataFrame, intercept: float, slope: float) -> float:
    sum: float = 0
    for row in data.values:
        sum += pow(row[1] - predict(row[0], intercept, slope), 2)
    return 1/len(data)*sum

def calculate_gradient(data: pd.DataFrame, intercept: float, slope: float, is_slope: bool = False) -> float:
    sum: float = 0
    for row in data.values:
        sum += predict(row[0], intercept, slope) - row[1] * row[0] if is_slope \
        else predict(row[0], intercept, slope) - row[1]
    return (1/len(data))*sum

def forward_propagation(data: pd.DataFrame, intercept: float, slope: float) -> float:
    return cost_function(data, intercept, slope)

def backward_propagation(data: pd.DataFrame, intercept: float, slope: float, learning_rate: float) -> Tuple[float, float]:
    intercept_gradient: float = calculate_gradient(data, intercept, slope, is_slope=False)
    slope_gradient: float = calculate_gradient(data, intercept, slope, is_slope=True)


    new_intercept: float = intercept - learning_rate * intercept_gradient
    new_slope: float = slope - learning_rate * slope_gradient
    return new_intercept, new_slope

def train(data: pd.DataFrame, intercept: float, slope: float, learning_rate: float, epochs: int) -> float:
    costs: List = []
    for i in range(30):
        costs.append(forward_propagation(data, intercept, slope))
        # print(costs[-1])
        intercept, slope = backward_propagation(data, intercept, slope, learning_rate)
        print(costs[-1], 'b=', intercept, 'm=', slope)
        save_model_parameters(intercept, slope)
    return 0.0

def main() -> None:
    intercept, slope = get_model_parameters()
    learning_rate, epochs = get_hyperparameters()
    print(intercept, slope, learning_rate, epochs)
    data: pd.DataFrame = parse_and_check_csv()
    data = normalize_data(data)
    train(data, intercept, slope, learning_rate, epochs)

if __name__ == '__main__':
    main()
