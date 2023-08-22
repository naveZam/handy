import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np


def create_data_plot():
    iris = datasets.load_iris()
    X = iris.data[:, 2:4]  # we only take the first two features.
    y = iris.target
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    plt.figure(2, figsize=(8, 6))
    plt.clf()

    colors = []
    for i in range(len(y)):
        if y[i] == 0:
            colors.append("red")
        if y[i] == 1:
            colors.append("green")
        if y[i] == 2:
            colors.append("blue")

    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolor='k')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    return X, y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def calc_h_theta(X, theta):
    return sigmoid(np.dot(X, theta.T))


def binary_to_types(y):
    types = np.zeros((3, len(y)))
    for i in range(len(y)):
        if y[i] == 0:
            types[0][i] = 1
        if y[i] == 1:
            types[1][i] = 1
        if y[i] == 2:
            types[2][i] = 1
    return types


def computeCost(X, y, theta):
    m = y[0].size
    grad_J = np.zeros(theta.shape)
    h_theta = np.zeros(y.shape)
    J = np.zeros(3)
    data = np.c_[np.ones((150, 1)), X]
    for i in range(len(theta)):
        h_theta[i] = (calc_h_theta(data, theta[i]))

    for k in range(len(J)):
        J[k] = np.sum(y[k] * np.log(h_theta[k]) + (1 - y[k]) * np.log(1 - h_theta[k]))   # Like for i in range(m)
        for i in range(m):
            for j in range(len(theta[k])):
                if j < 2:
                    grad_J[k][2 - j] += (h_theta[k][i] - y[k][i]) * X[i][1 - j]
                else:
                    grad_J[k][2 - j] += (h_theta[k][i] - y[k][i])


    J = - 1 / m * J
    grad_J = 1 / m * grad_J
    return J, grad_J


def computeCostCircular(X, y, theta):
    m = len(y)
    grad_J = np.zeros(theta.shape)
    J = np.zeros(3)
    data = np.c_[np.ones((150, 1)), X]
    h_theta = (calc_h_theta(data, theta.T))

    J = np.sum(y * np.log(h_theta) + (1 - y) * np.log(1 - h_theta))   # Like for i in range(m)
    for i in range(m):
        for j in range(len(theta)):
            if j < 2:
                grad_J[2 - j] += (h_theta[i] - y[i]) * X[i][1 - j]
            else:
                grad_J[2 - j] += (h_theta[i] - y[i])


    J = - 1 / m * J
    grad_J = 1 / m * grad_J
    return J, grad_J


def train(X, y, theta):
    J_values = []
    for i in range(1500):
        J, grad_J = computeCost(X, y, theta)
        theta[0] = theta[0] - 0.01 * grad_J[0]
        theta[1] = theta[1] - 0.1 * grad_J[1]
        theta[2] = theta[2] - 0.1 * grad_J[2]
        J_values.append(J)
    return theta, J_values


def circular_data(data):
    new_data = np.zeros((len(data), 5))
    for i in range(len(data)):
        new_data[i][0] = data[i][0]
        new_data[i][1] = data[i][1]
        new_data[i][2] = data[i][2]
        new_data[i][3] = data[i][1] ** 2
        new_data[i][4] = data[i][2] ** 2
    return new_data


def calculate_x2(x, theta):
    return -((theta[0] + theta[1] * x) / theta[2])


def calculate_circular_x2(x, theta):
    return theta[0] + theta[1] * x[0] + theta[2] * x[1] + theta[3] * x[0] ** 2 + theta[4] * x[1] ** 2


def test_train(X, y, theta):
    J_values = []
    for i in range(1000):
        J, grad_J = computeCostCircular(X, y, theta)
        theta = theta - 0.01 * grad_J
        J_values.append(J)
    return theta, J_values


def main():
    X, y = create_data_plot()
    theta_values = np.zeros((3, 3))
    theta_versicolor = np.ones((5, 1))
    data = circular_data(np.c_[y, X])
    data = np.c_[np.ones((150, 1)), np.delete(data, 0, axis=1)]

    h_theta = calc_h_theta(data, theta_versicolor.T)
    print(h_theta.T)

    y_types = binary_to_types(y)

    print(y_types[1])
    print(computeCostCircular(np.delete(data, 0, axis=1), y_types[1], theta_versicolor))
    theta_versicolor, J_values_test = test_train(np.delete(data, 0, axis=1), y_types[1], theta_versicolor)
    print(theta_versicolor)

    #print(X, y_types[0], theta_values[0], np.c_[X, y_types[0]])

    theta, J_values = train(X, y_types, theta_values)
    print(theta)
    x_values = np.linspace(0, 7, 100)  # Generate x values

    # Calculate x2 values using the function
    x2_values = [calculate_x2(x, theta[0]) for x in x_values]

    plt.plot(x_values, x2_values)
    x2_values = [calculate_x2(x, theta[1]) for x in x_values]

    plt.plot(x_values, x2_values)
    x2_values = [calculate_x2(x, theta[2]) for x in x_values]

    plt.plot(x_values, x2_values)
    plt.show()

    ys = J_values
    xs = [x for x in range(len(J_values))]
    plt.plot(xs, ys)
    plt.show()


    ys = J_values_test
    xs = [x for x in range(len(J_values_test))]
    plt.plot(xs, ys)
    plt.show()


    x = np.linspace(0, 7, 100)
    y = np.linspace(0, 7, 100)
    X, Y = np.meshgrid(x, y)

    plt.contour(X, Y, calculate_circular_x2([X, Y], theta_versicolor), [0])
    plt.show()


if __name__ == '__main__':
    main()