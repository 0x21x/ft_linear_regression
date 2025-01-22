from linear_regression import LinearRegression

if __name__ == '__main__':
    model = LinearRegression(training=False)
    print(f"Precision: {model.precision()}")
