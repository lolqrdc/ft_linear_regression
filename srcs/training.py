"""
Linear regression training program: trains a model on car mileage/price data.
"""
import csv
import os
import matplotlib.pyplot as plt


def read_data(data):
    km = []
    price = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, data)
    
    with open(filepath) as csvfile:
        datareader = csv.reader(csvfile)
        next(datareader)
        for row in datareader:
            km.append(float(row[0]))
            price.append(float(row[1]))
    return km, price


def mean(arr):
    return sum(arr) / len(arr)


def std(arr):
    m = mean(arr)
    variance = sum((x - m) ** 2 for x in arr) / len(arr)
    return variance ** 0.5


def normalize(arr):
    m = mean(arr)
    s = std(arr)
    return [(x - m) / s for x in arr]


def train_gradient_descent(x, y, alpha=0.01, epochs=1000):
    theta0 = 0
    theta1 = 0
    m = len(x)

    for _ in range(epochs):
        y_pred = [theta0 + theta1 * xi for xi in x]
        error = [y_pred[i] - y[i] for i in range(m)]

        theta0 -= alpha * (1/m) * sum(error)
        theta1 -= alpha * (1/m) * sum(error[i] * x[i] for i in range(m))
    return (theta0, theta1)


def main():
    x, y = read_data('data.csv')
    x_mean = mean(x)
    x_std = std(x)
    x_normalized = normalize(x)
    
    theta0, theta1 = train_gradient_descent(x_normalized, y)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, 'thetas.txt'), 'w') as f:
        f.write(f"{theta0}\n{theta1}\n{x_mean}\n{x_std}\n")
    
    print(f"Training complete!\nθ0 = {theta0}\nθ1 = {theta1}")
    
    y_pred = [theta0 + theta1 * xi for xi in x_normalized]
    
    plt.figure(figsize=(12, 7))
    plt.scatter(x, y, color='blue', label='Données réelles', alpha=0.6, s=50)
    plt.plot(x, y_pred, color='red', linewidth=2.5, label='Régression linéaire')
    plt.xlabel('Kilométrage (km)', fontsize=12)
    plt.ylabel('Prix (€)', fontsize=12)
    plt.title('Régression Linéaire : Kilométrage vs Prix', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'regression_plot.png'), dpi=300)
    print("✓ Graphique sauvegardé : regression_plot.png")
    plt.show()


if __name__ == "__main__":
    main()