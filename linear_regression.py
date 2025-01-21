import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Tuple, List
from utils import get_model_parameters, parse_and_check_csv, save_model_parameters

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def prediction(mileage: float, intercept: float, slope: float) -> float:
    """
        Predict the price of a car with the given mileage
    """
    return intercept + (mileage * slope)

def cost_function(data: pd.DataFrame, intercept: float, slope: float) -> float:
    """
        Calculate the cost function
    """
    sum: float = np.sum([(row[1] - prediction(row[0], intercept, slope)) ** 2 for row in data.values])
    return 1/len(data)*sum

def calculate_gradient(data: pd.DataFrame, intercept: float, slope: float, is_slope: bool = False) -> float:
    """
        Calculate the gradient of the cost function
    """
    sum: float = np.sum([(prediction(row[0], intercept, slope) - row[1]) * row[0] if is_slope \
        else prediction(row[0], intercept, slope) - row[1] for row in data.values])
    return 1/len(data)*sum

def forward_propagation(data: pd.DataFrame, intercept: float, slope: float) -> float:
    """
        Perform forward propagation
    """
    return cost_function(data, intercept, slope)

def backward_propagation(data: pd.DataFrame, intercept: float, slope: float, learning_rate: float) -> Tuple[float, float]:
    """
        Perform backward propagation by calculate the gradients and updating the parameters
    """
    intercept_gradient: float = calculate_gradient(data, intercept, slope, is_slope=False)
    slope_gradient: float = calculate_gradient(data, intercept, slope, is_slope=True)
    new_intercept: float = intercept - learning_rate * intercept_gradient
    new_slope: float = slope - learning_rate * slope_gradient
    return new_intercept, new_slope

class LinearRegression:
    def __init__(self: 'LinearRegression', training: bool = False) -> None:
        """
            Initialize the LinearRegression model
        """
        self.training: bool = training
        self.data: pd.DataFrame = parse_and_check_csv(normalized=True if self.training else False)
        self.intercept, self.slope, self.km_min, self.km_max, \
            self.price_min, self.price_max = get_model_parameters()
        if self.training:
            self.learning_rate: float = 1e-2
            
            self.epochs: int = 30000
            self.costs: List[float] = []
            self._normalize()

    def train(self: 'LinearRegression', with_plot: bool = False) -> None:
        """
            Train the model with the given dataset
        """
        if not self.training:
            logging.warning("Training unavailable without the training flag!")
            return
        for i in range(self.epochs):
            cost: float = forward_propagation(self.data, self.intercept, self.slope)
            self.costs.append(cost)
            self.intercept, self.slope = backward_propagation(self.data, self.intercept, self.slope, self.learning_rate)
            if i % 1000 == 0:
                logging.info(f"Epoch {i} - Cost: {cost}")
        self._denormalize()
        save_model_parameters(self.intercept, self.slope)
        if with_plot:
            self._plot_costs()

    def predict(self: 'LinearRegression', mileage: float) -> float:
        """
            Predict the price of a car with the given mileage
        """
        return prediction(mileage, self.intercept, self.slope)

    def plot(self: 'LinearRegression') -> None:
        """
            Plot the dataset
        """
        if self.training:
            logging.warning("Plotting only available without the training flag!")
            return
        plt.scatter(self.data['km'], self.data['price'])
        plt.plot(self.data['km'], self.intercept + self.slope * self.data['km'], color='red')
        plt.xlabel('Mileage')
        plt.ylabel('Price')
        plt.title('Dataset')
        plt.show()

    def _plot_costs(self: 'LinearRegression') -> None:
        """
            Plot the cost function
        """
        if not self.training:
            logging.warning("Plotting unavailable without the training flag!")
            return
        plt.plot(range(len(self.costs)), self.costs)
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.title('Cost function')
        plt.show()

    def _normalize(self: 'LinearRegression') -> None:
        """
            Normalize the model parameters
        """
        if not self.training:
            logging.warning("Normalization unavailable without the training flag!")
            return
        if self.intercept == 0 or self.slope == 0:
            return
        self.intercept = (self.intercept - self.price_min) / (self.price_max - self.price_min)
        self.slope = (self.slope - self.price_min) / (self.price_max - self.price_min)
        logging.info(f"Normalized parameters - Intercept: {self.intercept}, Slope: {self.slope}")

    def _denormalize(self: 'LinearRegression') -> None:
        """
            Denormalize the model parameters
        """
        if not self.training:
            logging.warning("Denormalization unavailable without the training flag!")
            return
        self.intercept = self.intercept * (self.price_max - self.price_min) + self.price_min
        self.slope = self.slope * (self.price_max - self.price_min) / (self.km_max - self.km_min)
        logging.info(f"Denormalized parameters - Intercept: {self.intercept}, Slope: {self.slope}")
