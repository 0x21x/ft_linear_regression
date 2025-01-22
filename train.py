from linear_regression import LinearRegression

def main() -> None:
    model = LinearRegression(training=True)
    model.train(with_plot=True)

if __name__ == '__main__':
    main()
