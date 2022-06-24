"""
Functions of problems 1.1, 1.2, 1.3

"""
import numpy as np


"""
The "noise" function selects n random points from a Gaussian normal distribution.
Parameters:
-mu: The mean of the Gaussian normal distribution
-sigma: The variance of the Gaussian normal distribution
-n: The number of random points to return
Returns:
-A list of n random points of a Gaussian normal distribution which are going to 
 be used as the noise
"""
def noise(mu, sigma, n):
    return np.random.normal(mu, sigma, n)


"""
The "y_vector" function finds the dot product of a parameter vector and an X vector.
Parameters:
-theta_vector: The parameter vector
-train_set: The X vector
Returns:
-A Y vector
"""
def y_vector(theta_vector, X_vector):
    return theta_vector.dot(X_vector.T)


"""
Implementation of the Least Squares method.
Parameters:
-train_X: The X vector of the training set
-train_y: The Y vector of the training set
Returns:
-A calculated parameter vector
"""
def least_squares(train_X , train_y):
    return np.linalg.inv(train_X.T.dot(train_X)).dot(train_X.T).dot(train_y)


"""
Calculation of the Mean Square Error.
Parameters:
-y1: The Y vector of the training set
-y2: The estimated Y vector
-axis: The axis on which the mean is going to be calculated (default: axis=0)
Returns:
-The Mean Square Error 
"""
def mse(y1, y2, axis = 0):
    return np.mean(((y1 - y2)**2), axis = axis)


"""
Implementation of the Ridge Regression function.
Parameters:
-train_X: The X vector of the training set
-train_y: The Y vector of the training set
-lamda: Tuning parameter of the ridge regression function - Lamda is a constant
Returns:
-A calculated parameter vector
"""
def ridge_regression(train_X , train_y, lamda):
    return np.linalg.inv(train_X.T.dot(train_X)+ lamda*(np.identity(train_X.shape[1]))).dot(train_X.T).dot(train_y)