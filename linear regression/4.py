import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import random



def gradiant_decent(weight, arrX, arrY):
    gWeight = [0, 0, 0]
    n = len(arrY)
    for i in range(n):
        x = arrX[i]
        y = arrY[i]
        gWeight[2] += -(2/n)* x[2] * singleCost(weight, x, y)
        gWeight[1] += -(2/n) * x[1] * singleCost(weight, x, y)
        gWeight[0] += -(2/n) * singleCost(weight, x, y)
    newWeight = updateWeights(weight, gWeight, 0.001)
    return newWeight

def updateWeights(newWeight, gWeight, learningRate):
    for i in range(len(newWeight)):
        newWeight[i] = newWeight[i] - gWeight[i] * learningRate
    return newWeight

def singleCost(weight, x, y):
    return (y - (weight[0] + weight[1] * x[1] + weight[2] * x[2]))

def cost(weight,data, y):
    cost = 0
    i = 0
    for a in data:
        cost+=  singleCost(weight, data, y[i]) ** 2
        i += 1
    return cost

costs = []
weight = [0, 0, 0]
bias = 0
epochs = 30
x = np.load("Exercise 1 data-20230826\\TA_Xhouses.npy")
y = np.load("Exercise 1 data-20230826\\TA_yprice.npy")

#SkLearn
poly = PolynomialFeatures(degree=2)
xpoly = poly.fit_transform(x)
model = LinearRegression(fit_intercept = True)
model.fit(xpoly, y)
xfit = np.linspace(0,15, 100)[:, np.newaxis]
xfitPoly = PolynomialFeatures.transform(poly, xfit)
yfit = model.predict(xfitPoly)
#SkLearn

new_x = []
for num in x:
    new_x.append([1, num[0], num[0] **2])
for i in range(epochs):
    print(weight)
    weight = gradiant_decent(weight, new_x, y)
    
plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()

plt.clf()
plt.scatter(x, y)
plt.plot(list(range(4, 12)), [weight[0] + weight[1] * i + weight[2] * i ** 2 for i in range(4, 12)])
plt.show()
