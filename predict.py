from utils import get_model_parameters
from train import prediction

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
    mileage: int = ask_mileage()
    intercept, slope, *_ = get_model_parameters()
    price: float = prediction(mileage, intercept, slope)
    print(f'The estimated price of the car is: {price}$')

if __name__ == '__main__':
    main()
