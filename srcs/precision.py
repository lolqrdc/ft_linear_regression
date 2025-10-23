"""
Model precision evaluation program: calculates accuracy metrics for the trained model.
"""
import os
import csv
import math


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

x = []
y = []
with open(os.path.join(script_dir, 'data.csv')) as csvfile:
    datareader = csv.reader(csvfile)
    next(datareader)
    for row in datareader:
        x.append(float(row[0]))
        y.append(float(row[1]))

x_normalized = [(xi - x_mean) / x_std for xi in x]
y_pred = [theta0 + theta1 * xi for xi in x_normalized]

m = len(x)

mae = sum(abs(y_pred[i] - y[i]) for i in range(m)) / m
mse = sum((y_pred[i] - y[i]) ** 2 for i in range(m)) / m
rmse = math.sqrt(mse)

y_mean = sum(y) / m
ss_tot = sum((y[i] - y_mean) ** 2 for i in range(m))
ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(m))
r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

print("\n" + "="*50)
print("📊 PRÉCISION DU MODÈLE")
print("="*50)
print(f"MAE (Erreur Absolue Moyenne) : {mae:.2f} €")
print(f"MSE (Erreur Quadratique Moyenne) : {mse:.2f}")
print(f"RMSE (Racine MSE) : {rmse:.2f} €")
print(f"R² (Coefficient de détermination) : {r_squared:.4f}")
print("="*50)

if r_squared >= 0.9:
    print("✅ Excellent fit!")
elif r_squared >= 0.7:
    print("✓ Good fit")
elif r_squared >= 0.5:
    print("⚠ Acceptable fit")
else:
    print("❌ Poor fit")
print("="*50 + "\n")
