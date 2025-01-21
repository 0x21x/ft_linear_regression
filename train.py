import numpy as np
import pandas as pd
from typing import Tuple, List
from matplotlib import pyplot as plt
from utils import get_model_parameters, get_hyperparameters, parse_and_check_csv, save_model_parameters, get_dataset_parameters

def prediction(mileage: float, intercept: float, slope: float) -> float:
    return intercept + (mileage * slope)

def cost_function(data: pd.DataFrame, intercept: float, slope: float) -> float:
    sum: float = 0
    for row in data.values:
        sum += (row[1] - prediction(row[0], intercept, slope)) ** 2
    return 1/len(data)*sum

def calculate_gradient(data: pd.DataFrame, intercept: float, slope: float, is_slope: bool = False) -> float:
    sum: float = np.sum([(prediction(row[0], intercept, slope) - row[1]) * row[0] if is_slope \
        else prediction(row[0], intercept, slope) - row[1] for row in data.values])
    return 1/len(data)*sum

def forward_propagation(data: pd.DataFrame, intercept: float, slope: float) -> float:
    return cost_function(data, intercept, slope)

def backward_propagation(data: pd.DataFrame, intercept: float, slope: float, learning_rate: float) -> Tuple[float, float]:
    intercept_gradient: float = calculate_gradient(data, intercept, slope, is_slope=False)
    slope_gradient: float = calculate_gradient(data, intercept, slope, is_slope=True)
    new_intercept: float = intercept - learning_rate * intercept_gradient
    new_slope: float = slope - learning_rate * slope_gradient
    return new_intercept, new_slope

def plot(costs: List) -> None:
    plt.plot(costs)
    plt.show()

def train(data: pd.DataFrame, intercept: float, slope: float, learning_rate: float, epochs: int) -> float:
    costs: List = []
    km_min, km_max, price_min, price_max = get_dataset_parameters()
    for i in range(epochs):
        cost: float = forward_propagation(data, intercept, slope)
        costs.append(cost)
        intercept, slope = backward_propagation(data, intercept, slope, learning_rate)
    slope = slope * (price_max - price_min) / (km_max - km_min)
    intercept = intercept * (price_max - price_min) + price_min - slope * km_min
    save_model_parameters(intercept, slope)
    plot(costs)
    return 0.0

def main() -> None:
    dataset: pd.DataFrame = parse_and_check_csv()
    intercept, slope, km_min, km_max, price_min, price_max = get_model_parameters()
    print(f"intercetp: {intercept}, slope: {slope}")
    learning_rate, epochs = get_hyperparameters()
    train(dataset, intercept, slope, learning_rate, epochs)

if __name__ == '__main__':
    main()
