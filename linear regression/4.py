import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import random
DEGREE = 2


def gradiant_decent(weight, arrX, arrY):
    gWeight = np.zeros(DEGREE + 1)
    print(gWeight)
    n = len(arrY)
    for i in range(n):
        x = arrX[i]
        y = arrY[i]
        
        gWeight[2] += x[2] * (y - np.dot(x, weight))
        gWeight[1] += x[1] * (y - np.dot(x, weight))
        gWeight[0] += (y - np.dot(x, weight))
    gWeight *= -(2/n)
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
        cost+=  weight @ x ** 2
        i += 1
    return cost

costs = []
weight = np.ones(3)
bias = 0
epochs = 30
x = np.load("C:\\Users\\orile\\Downloads\\Exercise 1 data-20230826\\TA_Xhouses.npy")
y = np.load("C:\\Users\\orile\\Downloads\\Exercise 1 data-20230826\\TA_yprice.npy")

#SkLearn
poly = PolynomialFeatures(degree=2)
xpoly = poly.fit_transform(x)
model = LinearRegression(fit_intercept = True)
model.fit(xpoly, y)
xfit = np.linspace(0,15, 100)[:, np.newaxis]
xfitPoly = PolynomialFeatures.transform(poly, xfit)
yfit = model.predict(xfitPoly)
#SkLearn
i = 0
new_x = []
for num in x:
    
    new_x.append([np.float64(1)])
    for j in range(DEGREE):
        new_x[i].append(num ** (j +1))
    i += 1
print(new_x)
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
