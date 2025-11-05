# Bonus : a program that calculate the precision of my algorithm

import os
import csv
import math

# Reload data and params
def loadParams(filename="saved.txt"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)
    try:
        with open(filepath, 'r') as f:
            theta0 = float(f.readline())
            theta1 = float(f.readline())
            x_mean = float(f.readline())
            x_std = float(f.readline())
        return (theta0, theta1, x_mean, x_std)
    except FileNotFoundError:
        print(f"Error: {filename} not found. Run training.py first.")
        exit(1)
    except ValueError:
        print(f"Error: corrupted file {filename}.")
        exit(1)

def loadData(filename="data.csv"):
    x = []
    y = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)
    with open(filepath, newline='') as csvfile:
        datareader = csv.reader(csvfile)
        next(datareader)
        for row in datareader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    return(x, y)

def predict(theta0, theta1, mean_x, std_x, x_values):
    x_norm = [(xi - mean_x) / std_x for xi in x_values]
    return([theta0 + theta1 * xi for xi in x_norm])

def calcMetrics(y_true, y_pred):
    n = len(y_true)
    mae = sum(abs(y_pred[i] - y_true[i]) for i in range(n)) / n
    mse = sum((y_pred[i] - y_true[i])**2 for i in range(n)) / n
    rmse = math.sqrt(mse)

    y_mean = sum(y_true) / n
    ss_tot = sum((y_true[i] - y_mean)**2 for i in range(n)) / n
    ss_res = sum((y_true[i] - y_pred[i])**2 for i in range(n))
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    return (mae, mse, rmse, r2)

def main():
    theta0, theta1, mean_x, std_x = loadParams()
    x, y = loadData()
    y_pred = predict(theta0, theta1, mean_x, std_x, x)
    mae, mse, rmse, r2 = calcMetrics(y, y_pred)

    print("PRECISION OF THE CHOOSED ALGORITHM")
    print(f"MAE (Erreur absolue moyenne): {mae:.2f} €")
    print(f"MSE (Erreur quadratique moyenne): {mse:.2f}")
    print(f"RMSE (Racine MSE): {rmse:.2f} €")
    print(f"R² (Coefficient de determination): {r2:.4f}")

if __name__ == "__main__":
    main()
