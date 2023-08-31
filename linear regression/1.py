import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.linear_model import LinearRegression
# preparing the data
a = 2.5
b = 10
x = np.random.rand(500) * 50
y = b + a * x + np.random.normal(0, 25, 500)
plt.scatter(x,y)

model = LinearRegression(fit_intercept = True)
model.fit(x[:, np.newaxis], y)
xfit = np.linspace(0,50, 10000)
yfit = model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()
print("Model slope a1 = ", model.coef_[0])
print("Model intercept a0 = ", model.intercept_)
print(x.tolist())
print(y)
