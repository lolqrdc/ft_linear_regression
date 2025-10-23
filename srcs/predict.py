"""
Car price prediction program: uses trained linear regression model to estimate prices.
"""
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(script_dir, 'thetas.txt'), 'r') as f:
        theta0 = float(f.readline())
        theta1 = float(f.readline())
        x_mean = float(f.readline())
        x_std = float(f.readline())
except FileNotFoundError:
    print("Error: thetas.txt not found. Run training.py first!")
    exit(1)

try:
    mileage = float(input("Enter a mileage: "))
    
    if mileage < 0:
        print("Error: Mileage cannot be negative!")
        exit(1)
        
except ValueError:
    print("Error: Please enter a valid number")
    exit(1)

mileage_normalized = (mileage - x_mean) / x_std
estimated_price = theta0 + theta1 * mileage_normalized

if estimated_price < 0:
    print(f"Estimated price: Price too low (negative value: {estimated_price:.2f} €)")
else:
    print(f"Estimated price: {estimated_price:.2f} €")