import numpy as np

def load_dataset(filename): 
    dataset = np.genfromtxt(filename, dtype='float', delimiter=',')
    dataset = np.array(dataset)
    class_labels = np.unique(dataset[:, -1])

    return dataset, class_labels, len(dataset[0,:-1])


"""
Finds the mean and the covariance matrices(diagonal, with all diagonal elements equal) for Question (a)
Parameters:
-data_set: The splitted dataset
Returns:
-mean: The mean vector
-covar_matrix: The covariance matrix
"""
def mean_var_a(data_set):
    mean = np.mean(data_set,axis=0)

    dif = (data_set - mean)
    var = np.sum(np.dot(dif.T, dif)) / (data_set.shape[1] * data_set.shape[0])

    covar_matrix = var*np.identity(len(data_set[0]))

    return mean, covar_matrix

"""
Finds the mean and the covariance matrices (non-diagonal).
Means and covariance matrices of the pdfs are estimated using Maximum Likelihood 
from the available data. For Question (b)
Parameters:
-data_set: The splitted dataset
Returns:
-mean: The mean vector
-covar_matrix: The covariance matrix
"""
def mean_var_b(data_set):
    mean = np.mean(data_set,axis=0)
    
    dif = (data_set - mean)
    covar_matrix = np.dot(dif.T, dif) / data_set.shape[0]

    return mean, covar_matrix


"""
Finds the mean and the covariance matrices.
Components of the feature vectors are mutually statistically independent
Marginal Pdfs are gaussian, with parameters (mean, variance) estimated using Maximum Likelihood 
from the available data. For Question (c)
Parameters:
-data_set: The splitted dataset
Returns:
-mean: The mean vector
-covar_matrix: The covariance matrix
"""
def mean_var_c(data_set):
    mean = np.mean(data_set,axis=0)

    dif = (data_set - mean)

    var = np.var(data_set,axis=0)
    covar_matrix = var*np.identity(len(data_set[0]))

    return mean, covar_matrix

"""
Finds the mean and the covariance matrices.
Components of the feature vectors are mutually statistically independent
Marginal Pdfs are gaussian, with parameters (mean, variance) estimated using Maximum Likelihood 
from the available data. For Question (c)
Parameters:
-data_set: The splitted dataset
-x: A row of the data_set
Returns:
-parz: A probability
"""
def parzen(data_set,x): 
    
    # Number of rows
    N = len(data_set)
    
    h=np.sqrt(N)

    # Calculates the pd for all points of data_set, for each dimension separately, using as mean the x 
    # and as variance the h^2:
    p = gaussian(data_set, x, h**2, 1)

    # Sums the pd's over all points of the data_set
    parz = np.sum(p,axis=0)/N
    
    # Multiplies the pd's of all the dimensions together to get the final pd of x
    parz = np.prod(parz)

    return parz


"""
The gaussian function calculates the pd for all points of a data set, for each dimension separately.
Parameters:
-dataset
-mean: Mean vector
-var: Covariance matrix
-dimension: If dimensions = 1 then implement for 1D Parzen
Results:
-prob: Probability
"""
def gaussian(dataset, mean = None, var = None, dimension = None):
    # calculates the pd for univariate gaussian distribution:
    if dimension == 1:
        nom = np.exp((-(dataset-mean)**2)/(2*var))
        denom = (2 * np.pi * var)**0.5 
        prob = nom / denom
    
    # calculates the pd for multivariate gaussian distribution:
    else:
        num_params = len(mean)
        nom = np.exp( -0.5 * (((dataset - mean) @ np.linalg.inv(var)) @ (dataset - mean).T))
        denom = ((2* np.pi) ** (num_params/2)) * (np.linalg.det(var) ** 0.5)
        prob = nom / denom     
        
    return prob