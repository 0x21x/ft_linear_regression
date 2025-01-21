from linear_regression import LinearRegression

def main() -> None:
    linear_regression = LinearRegression(training=True)
    linear_regression.train(with_plot=True)

if __name__ == '__main__':
    main()
