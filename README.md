# Linear Regression Model for Car Price Prediction

This project implements a simple linear regression model to predict car prices based on mileage. It's designed to demonstrate the basics of machine learning and data analysis using Python.

## 📊 Project Overview

The project consists of several Python scripts that work together to train a linear regression model, make predictions, and visualize the results.

### Key Components:

- **linear_regression.py**: Contains the core LinearRegression class and associated functions.
- **train.py**: Script to train the model.
- **predict.py**: Script to make predictions using the trained model.
- **plot.py**: Script to visualize the data and model results.
- **utils.py**: Utility functions for data processing and file operations.

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/ft_linear_regression.git
   cd ft_linear_regression
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## 🔧 Usage

### Training the Model

To train the model, run:

```
python train.py
```

This will process the data from `data.csv`, train the linear regression model, and save the model parameters.

### Making Predictions

To make predictions using the trained model, run:

```
python predict.py
```

Follow the prompts to enter the mileage of a car, and the script will output the predicted price.

### Visualizing the Data

To visualize the dataset and the regression line, run:

```
python plot.py
```

This will display a scatter plot of the data points along with the fitted regression line.

## 📁 Project Structure

```
ft_linear_regression/
│
├── data.csv                 # Input data file
├── linear_regression.py     # Main linear regression implementation
├── plot.py                  # Data visualization script
├── predict.py               # Prediction script
├── train.py                 # Model training script
├── utils.py                 # Utility functions
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## 🙏 Acknowledgements

- This project was created as part of the 42 school curriculum.

