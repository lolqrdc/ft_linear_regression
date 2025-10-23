# Linear Regression - Car Price Prediction

A simple linear regression implementation for predicting car prices based on mileage using gradient descent.

## Project Overview

This project implements a linear regression model with a single feature (car mileage) to estimate vehicle prices. The implementation uses manual gradient descent to calculate optimal theta values without relying on external ML libraries.

**Formula:** `estimatePrice(mileage) = θ0 + (θ1 × mileage)`

## Project Structure

```
ft_linear_regression/
├── srcs/
│   ├── training.py        # Model training with gradient descent
│   ├── predict.py         # Price prediction program
│   ├── precision.py       # Model evaluation metrics
│   ├── data.csv           # Training dataset
│   ├── thetas.txt         # Saved model parameters
│   └── regression_plot.png # Visualization of regression line
└── README.md
```

## Requirements

- Python 3.7+
- matplotlib (for visualization)

### Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install matplotlib
```

## Usage

### Step 1: Train the Model

```bash
python3 srcs/training.py
```

This script:
- Reads the training dataset from `data.csv`
- Normalizes the mileage data
- Performs gradient descent to calculate θ0 and θ1
- Saves parameters to `thetas.txt`
- Generates a regression plot visualization

**Output:**
```
Training complete!
θ0 = 6331.56
θ1 = -1105.97
✓ Graphique sauvegardé : regression_plot.png
```

### Step 2: Make Predictions

```bash
python3 srcs/predict.py
```

Enter a mileage value to get an estimated price:
```
Enter a mileage: 10000
Estimated price: 8284.75 €
```

### Step 3: Evaluate Model Performance

```bash
python3 srcs/precision.py
```

This displays accuracy metrics:
```
==================================================
📊 PRÉCISION DU MODÈLE
==================================================
MAE (Erreur Absolue Moyenne) : 557.83 €
MSE (Erreur Quadratique Moyenne) : 445645.32
RMSE (Racine MSE) : 667.57 €
R² (Coefficient de détermination) : 0.7330
==================================================
✓ Good fit
==================================================
```

## Implementation Details

### Gradient Descent Algorithm

The model uses the following update formulas:

```
tmpθ0 = learningRate × (1/m) × Σ(estimatePrice[i] - price[i])
tmpθ1 = learningRate × (1/m) × Σ((estimatePrice[i] - price[i]) × mileage[i])
```

Where:
- `m` = number of training examples
- `learningRate` = 0.01 (default)
- `epochs` = 1000 (iterations)

Both theta values are updated simultaneously in each iteration.

### Data Normalization

Input features are normalized using z-score normalization:
```
x_normalized = (x - mean(x)) / std(x)
```

This ensures stable convergence and prevents numerical overflow.

## Evaluation Metrics

- **MAE**: Mean Absolute Error - Average absolute difference between predictions and actual values
- **MSE**: Mean Squared Error - Average squared difference
- **RMSE**: Root Mean Squared Error - Standard deviation of errors
- **R²**: Coefficient of Determination - Proportion of variance explained (0-1 scale)

## File Descriptions

### training.py
Implements the full training pipeline including data loading, normalization, gradient descent, and visualization.

### predict.py
Loads trained parameters and predicts prices for user-provided mileage values with input validation.

### precision.py
Calculates and displays model performance metrics using trained parameters and dataset.

## Notes

- θ0 and θ1 are initialized to 0 before training
- Data must be in CSV format with headers
- Negative mileage values are rejected
- Negative price predictions are flagged as warnings
- The model respects project constraints (no numpy.polyfit, manual implementation of all calculations)

## Results

The model achieves good predictive performance with R² ≈ 0.73, explaining approximately 73% of the price variance based on mileage alone.
