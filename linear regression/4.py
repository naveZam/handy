import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import random
DEGREE = 2


def gradiant_decent(weight, arrX, arrY):
    gWeight = np.zeros(DEGREE + 1)
    n = len(arrY)
    
    gWeight = -(2/n) * np.dot(arrX.T, (arrY - np.dot(arrX, weight)))
    print(weight)
    print(gWeight)
    newWeight = weight - gWeight * 0.00001
    return newWeight


def cost(weight,arrX, arrY):
    return sum(arrY - np.dot(arrX, weight)) ** 2

def guess(weight, x):
    print(x)
    y = np.zeros(len(x))
    print(y)
    for i in range(len(x)):
        np.insert(y[i],   weight[0] + weight[1] * x + weight[2] * x*x, i)
    return y

costs = []
weight = np.ones(3)
bias = 0
epochs = 30000
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
new_x = np.ones((len(x)))
for i in range(1, DEGREE +1):
    new_x = np.c_[new_x, x**i]
print(new_x)
arrY = []
for i in range(len(y)):
    arrY.append(y[i][0])
y = np.array(arrY)
for i in range(epochs):
    weight = gradiant_decent(weight, new_x, y)
    #print(cost(weight, new_x, y))
    
plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()
 
plt.clf()
plt.scatter(x, y)
plt.plot(list(range(4, 16)), [weight[0] + weight[1] * i + weight[2] * i ** 2 for i in range(4, 16)])
plt.show()
