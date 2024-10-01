import numpy as np

from scipy import stats,optimize, linalg
from scipy.spatial.distance import cdist, pdist, squareform
# from sklearn.gaussian_process.kernels import RBF



import math

import os

from numpy import  median, shape, sqrt, ndarray

import pandas as pd
import pickle


def file_name(data_type = 'results',n = 50, simu = 'ii', nb_perm = 1000,nexp = 1000, alpha = 0.05, method = 'opti', estimethod = None):
    '''
    data_type: 'simus' or 'results', data simulations or results 
    '''
    save_path = './outputs/'+data_type+'/n'+str(n)+'/simu_'+simu
    if data_type == 'simus':
        name = save_path+'/simu_'+simu+'_nexp_'+str(nexp)+'.pkl'
    else:
        name = save_path+'/HSIC_'+method+'_B_'+str(nb_perm)+'_nexp_'+str(nexp)+'.pkl'
    return name

def load_data(file_data):
     '''
     load data contained in file_data (must be a .pkl)
     '''
     unpickle = open(file_data,'rb')
     df = pickle.load(unpickle)
     return df


# kernel width using median trick
def set_width_median(X: ndarray):
    n = np.shape(X)[0]
    if n > 1000:
        X = X[np.random.permutation(n)[:1000], :]
    dists = squareform(pdist(X, 'euclidean'))
    median_dist = np.median(dists[dists > 0])
    width = np.sqrt(2.) * median_dist
    theta = 1.0 / (width ** 2)
    width = theta
    return width


def set_bandwidth(data, method):
    '''
    set kernel width parameters in funciton of the method:
    - data = 1d-array
    - method : 'empirical' or 'median'
    '''
    data = data.reshape((len(data),1))
    if method == 'empirical':
        n = np.shape(data)[0]
        if n <= 200:
            width = 0.9
        elif n> 1200:
            width = 0.3
        else:
            width =0.5
    if method == 'median':
        width = set_width_median(data)
    return width

def center_matrix(K):
    '''
    Do not center with HKH with H = I -1/n 11^T (complexity O(n^3))
    --> use sums: 
    Notice that K (both Lx and Ly) are symmetric matrices, so K_colsums == K_rowsums
    '''
    n = np.shape(K)[0]
    K_colsums = K.sum(axis=0)
    K_allsum = K_colsums.sum()
    return K - (K_colsums[None, :] + K_colsums[:, None]) / n + (K_allsum / n ** 2)


def generate_unique_permutations(var, B):
    '''
    generate B unique permutations of variable var
    '''
    n = len(var)
    if B > math.factorial(n):
        print('nb permutations greater than n!')
        return 0
    
    unique_permutations = set() ## cheat code to get unique samples: set() removes one element if the same is added into the set
    
    while len(unique_permutations) < B:
        perm = tuple(np.random.permutation(var))
        unique_permutations.add(perm)
    
    return np.array(list(unique_permutations))

def compute_Gram_matrix(param,data):
    '''
    Compute kernel matrix for data with RBF :
    L_x = [kx(x_i,y_j)]_{1\leq i,j\leq n}
    param = bandwidth 1-D array
    '''
    kwx = param
    X = data.reshape((data.size,1))
    
    #compute kernels k_x 
    #if self.kx == 'Gaussian':
    ## compute distance in norm2 between each data points
    distx = cdist(X, X, 'sqeuclidean')
    K = np.exp(-0.5 * distx * kwx**-2) #gaussian kernel k(x1,x2) = exp(- 0.5 *||x1-x2||^2 * \sigma_x^-2)

    # #if self.ky == 'Gaussian':
    # ## compute distance in norm2 between each data points
    # disty = cdist(Y,Y, 'sqeuclidean')            
    # Ly = np.exp(-0.5 * disty * kwy**-2) #gaussian kernel k(y1,y2) = exp(- 0.5 *||y1-y2||^2 * \sigma_y^-2)

    return K


def det_with_lu(K):
    '''
    compute determinant of matrix K with LU decomposition
    '''
    P, L, U = linalg.lu(K)
    det = np.prod(np.diag(U)) * np.linalg.det(P)
    return det


def save_data(folder, param, X , Y, simu_type = None):
    '''
    save data from simulations:
    simu_type = 'i', 'ii', 'iii', 'ts'
    if simu_type = 'i' or 'ii', the parameter is l = int
    if simu_type = 'iii', the parameter is rho = float
    if simu_type = 'ts', parameters is a list [alpha,beta]
    '''
    ## check if folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    if simu_type == 'i' or simu_type == 'ii':
        df = pd.DataFrame(columns = ['l','X','Y'], index = range(1))
        df['X'].iloc[0] = X
        df['Y'].iloc[0] = Y
        df['l'] = param
    elif simu_type== 'iii':
        df = pd.DataFrame(columns = ['rho','X','Y'], index = range(1))
        df['X']= X
        df['Y'] = Y
        df['rho'] = param
    elif simu_type == 'ts':
        
        df = pd.DataFrame({'alpha': param[0], 'beta': param[1], 'X': X, 'Y': Y})
        
    n = len(X)
    
    df.to_pickle(folder+'n'+str(n)+'_simu_'+str(simu_type)+'.pkl')