# Train a linear regression model by using gradient descent

import csv
import os
import matplotlib.pyplot as plt

# Creation of a function to read the file data.csv, extract km & price, 
# and return those values.
def readData(data):
    km = []
    price = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, data)

    with open(filepath, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        next(datareader)
        for row in datareader:
            km.append(float(row[0]))
            price.append(float(row[1]))
    return (km, price);

# Initialize coefficients theta0 and theta1, run the gradient descent, 
# update parameters by using errors and save the final coefficients.
def gradientDescent(x, y, alpha=0.01, iterations=1000):
    theta0 = 0
    theta1 = 0
    m = len(x)

    for _ in range(iterations):
        tmp0 = 0
        tmp1 = 0
        for i in range(m):
            error = (theta0 + theta1 * x[i]) - y[i]
            tmp0 += error
            tmp1 += error * x[i]

        theta0 = theta0 - alpha * (tmp0 / m)
        theta1 = theta1 - alpha * (tmp1 / m)
    return (theta0, theta1);

# Useful functions to: calc mean, normalize data, and standard deviation.
def mean(arr):
    return (sum(arr)/len(arr))

def std(arr):
    m = mean(arr)
    return (sum((x - m) ** 2 for x in arr) / len(arr)) ** 0.5

def normalize(arr):
    m = mean(arr)
    s = std(arr)
    return ([(x-m)/s for x in arr])

# Save the coefficients to use it for predict.py.
def save_params(theta0, theta1, mean_x, std_x, filename="saved.txt"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)
    with open(filepath, "w") as f:
        f.write(f"{theta0}\n{theta1}\n{mean_x}\n{std_x}\n")

# Main 
def main():
    x, y = readData("data.csv")
    x_mean = mean(x)
    x_std = std(x)
    x_norm = normalize(x)

    theta0, theta1 = gradientDescent(x_norm, y, alpha=0.01, iterations=1000)
    save_params(theta0, theta1, x_mean, x_std, filename="saved.txt")

    # Visualisation
    y_pred = [theta0 + theta1 * xi for xi in x_norm]
    plt.scatter(x, y, color="blue", label="Data points")
    plt.plot(x, y_pred, color="red", label="Regression line")
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.legend()
    plt.title("Linear Regression")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, "regression_plot.png")
    plt.savefig(filepath)
    print(f"Graph saved to {filepath}")
    plt.close()

if __name__ == "__main__":
    main()

