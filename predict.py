from linear_regression import LinearRegression
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def ask_mileage() -> int:
    while True:
        try:
            mileage: int = int(input('Enter the mileage of the car: '))
            if mileage < 0:
                logging.warning('The mileage must be a positive number.')
            else:
                return mileage
        except ValueError:
            logging.error('Please enter a valid number.')

def main() -> None:
    model = LinearRegression()
    price: float = model.predict(ask_mileage())
    logging.info(f'The estimated price of the car is: {price}$')

if __name__ == '__main__':
    main()
