"""
Functions of problems 1.4, 1.5
"""
import numpy as np
import numpy.linalg


"""
The "get_Phi" calculates the Phi matrix.
Parameters:
-x: Initial X vector
-dp: Polynomial degrees
-N: Number of points
Returns:
-The Phi matrix of shape N x dp+1
"""
def get_Phi(x, dp, N):
    phi_pol = x
    for i in range(2,dp+1):
       phi_pol = np.c_[phi_pol, x**(i)]    #add the 5 polynomial degrees of each xi 
    phi_matrix = np.c_[np.ones((N,1)), phi_pol]   #add 1 for x0
    return phi_matrix


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
The "get_y_vector" function finds the dot product of a parameter vector and a Phi vector.
Parameters:
-theta_vector: The parameter vector
-train_set: The Phi vector
-mu_noise: The mean of the noise distribution
-noise_var: The variance of the noise distribution
-N: The number of random points to return
Returns:
-y: The Y vector without added noise
-y_noise: The Y vector with added noise
"""
def get_y_vector(theta_vector, train_set, mu_noise, noise_var, N):
    y = theta_vector.dot(train_set.T)
    y_noise = y + noise(mu_noise, noise_var**0.5, N)
    return y, y_noise


"""
Calculate the mean theta vector.
Parameters:
-noise_var: The variance of the noise distribution
-theta_var: The theta variance
-theta_0: The theta vector
-train_set: The Phi matrix
-y_train: The Y vector of the training set
-id: Identity matrix
Returns:
-The estimated parameter vector
"""
def fun_mu_th_y(noise_var, theta_var, theta_0, train_set, y_train, id):
    first_part = (1 / noise_var) * np.linalg.inv((1 / theta_var) * id + (1 / noise_var) * train_set.T.dot(train_set))
    second_part = train_set.T.dot(y_train - train_set.dot(theta_0))
    return theta_0 + first_part.dot(second_part)


"""
Calculates the mean and the variance of the estimated Y vector
Parameters:
-x_test: The X vector of the test set
-dp: Polynomial degrees
-N: Number of points
-mu_th_y: The estimated parameter vector
-noise_var: Variance of the noise
-theta_var: The theta variance
-train_set: The Phi matrix
Returns:
-mu_y: A list of the means of the estimated Y points 
-sigma_y = A list of the variances of the estimated Y points
"""
def fun_mu_sigma_y(x_test, dp, N, mu_th_y, noise_var, theta_var, train_set):
    phi_matrix_test = get_Phi(x_test, dp, N)
    
    mu_y = np.sum(phi_matrix_test * mu_th_y, axis=1)

    first_part = noise_var * theta_var * phi_matrix_test 
    second_part = np.linalg.inv(noise_var * np.identity(phi_matrix_test.shape[1]) + theta_var * train_set.T.dot(train_set))


    sigma_y = noise_var + np.sum((first_part @ second_part) * (phi_matrix_test), axis=1)

    return mu_y, sigma_y