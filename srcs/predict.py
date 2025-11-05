# Predict price from given mileage using the trained model

import os
# Loads data (theta0, theta1, mean_x, std_x) saved from "saved.txt".
def loadData(filename="saved.txt"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(script_dir, filename), 'r') as f:
            theta0 = float(f.readline())
            theta1 = float(f.readline())
            mean_x = float(f.readline())
            std_x = float(f.readline())
        return (theta0, theta1, mean_x, std_x)
    except FileNotFoundError:
        print("Error: saved.txt not found, please run training.py first.")
        exit(1)
    except ValueError:
        print("Error: corrupted file.")
        exit(1)

# Predict car price from mileage using normalized linear regression formula.
def predictPrice(mileage, theta0, theta1, mean_x, std_x):
    if mileage < 0:
        raise ValueError("Mileage cannot be negative")
    normalized = (mileage - mean_x) / std_x
    price = theta0 + theta1 * normalized
    return (price)

# Main
def main():
    theta0, theta1, mean_x, std_x = loadData()

    mileage_input = input("Enter mileage: ")
    try:
        mileage = float(mileage_input)
        price = predictPrice(mileage, theta0, theta1, mean_x, std_x)
        if price < 0:
            print(f"Estimated price too low (negative): {price:.2f} €")
        else:
            print(f"Estimated price: {price:.2f} €")
    except ValueError as e:
        print("Invalid input:", e)

if __name__ == "__main__":
    main()