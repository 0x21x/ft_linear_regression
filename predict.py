from utils import get_model_parameters

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

def predict(mileage: int) -> float:
    intercept, slope = get_model_parameters()
    return intercept + slope * mileage

def main() -> None:
    mileage: int = ask_mileage()
    price: float = predict(mileage)
    print(f'The estimated price of the car is: {price}$')

if __name__ == '__main__':
    main()
