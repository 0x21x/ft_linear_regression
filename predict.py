from linear_regression import LinearRegression

def ask_mileage() -> int:
    while True:
        try:
            mileage: int = int(input('Enter the mileage of the car: '))
            if mileage < 0:
                print('The mileage must be a positive number.')
            else:
                return mileage
        except ValueError:
            print('Please enter a valid number.')

def main() -> None:
    linear_regression = LinearRegression()
    price: float = linear_regression.predict(ask_mileage())
    print(f'The estimated price of the car is: {price}$')

if __name__ == '__main__':
    main()
