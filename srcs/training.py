# GOAL : train a linear regression model by using gradient descent

import csv
import os
import matplotlib.pyplot as plt

# Creation of a function to read the file data.csv, extract km & price, 
# and return those values.
def readData():
    km = []
    price = []

    with open('data.csv', newline='') as csvfile:
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

