import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
from sklearn.linear_model import LinearRegression
import random


def cost(weight, bias, data, y):
    cost = 0
    i = 0
    for a in data:
        cost += (y[i] - (a * weight + bias)) ** 2
        i += 1
    return cost

def gradiant_decent(weight, bias, arrX, arrY):
    gWeight = 0
    gBias = 0
    n = len(arrX)
    for i in range(n):
        x = arrX[i]
        y = arrY[i]
        gWeight += -(2/n) * x * (y - (weight * x + bias))
        gBias += -(2 / n) * (y - (weight * x + bias))
    newWeight = weight - gWeight * 0.001
    newBias = bias - gBias * 0.001
    return newBias, newWeight
x = []
y = []
weight = 0
bias = 0
epochs = 30
costs = []
data = np.load("C:\\Users\\orile\\Downloads\\Exercise 1 data-20230826\\Cricket.npz")
for point in data["arr_0"]:
    x.insert(len(x), point[0])
    y.insert(len(x), point[1])


for i in range(epochs):
    batchX = []
    batchY = []
    numbers = set()
    while len(numbers) < 5:
        num = random.randint(0, 13)
        numbers.add(num)
    for num in numbers:
        batchY.append(y[num])
        batchX.append(x[num])
    costs.append(cost(weight, bias, x, y))
    bias, weight = gradiant_decent(weight, bias, batchX, batchY)
print(weight, bias)
plt.scatter(x, y)
plt.plot(list(range(14, 22)), [weight * x + bias for x in range(14, 22)])
plt.show()

plt.clf()
plt.scatter(range(0, 30), costs)
plt.show()

print("The y for the x 85 58 and 39 are:")
print(85 * weight + bias)
print(58 * weight + bias)
print(39 * weight + bias)
