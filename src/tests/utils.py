import numpy as np

from scipy import stats,optimize, linalg
from scipy.spatial.distance import cdist, pdist, squareform
# from sklearn.gaussian_process.kernels import RBF



import math

import os



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
def set_width_median(X: np.ndarray):
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

# def center_matrix(K):
    '''
    Do not center with HKH with H = I -1/n 11^T (complexity O(n^3))
    --> use sums: 
    Notice that K (both Lx and Ly) are symmetric matrices, so K_colsums == K_rowsums
    '''
    n = np.shape(K)[0]
    K_colsums = K.sum(axis=0)
    K_allsum = K_colsums.sum()
    return K - (K_colsums[None, :] + K_colsums[:, None]) / n + (K_allsum / n ** 2)



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

def generate_permuted_matrices(M, B):
    '''
    Generate B permuted matrices of size (n \times n) 
    '''
    n = M.shape[0]
    permuted_samples = []

    for _ in range(B):
        permuted_M = M[:, np.random.permutation(n)]
        permuted_samples.append(permuted_M)

    return permuted_samples

def center_matrix(A):
    ### Double center the matrix A
    n, m = A.shape
    S_mat = np.sum(A) / (n * m)
    S_rows = np.sum(A, axis=1) / m
    S_cols = np.sum(A, axis=0) / n

    B = A - np.outer(S_rows, np.ones(m)) - np.outer(np.ones(n), S_cols) + S_mat
    return B


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
    Compute kernel matrix for data with RBF kernel:
    L_x = [kx(x_i,x_j)]_{1\leq i,j\leq n}
    param = bandwidth parameter
    '''
    kwx = param
    X = data.reshape((data.size,1))
    
    ## compute distance in norm2 between each data points
    distx = cdist(X, X, 'sqeuclidean')
    K = np.exp(-0.5 * distx * kwx**-2) #gaussian kernel k(x1,x2) = exp(- 0.5 *||x1-x2||^2 * \sigma_x^-2)

    return K

def meannondiag(A):
    """
    Compute the mean of the non-diagonal elements of a square matrix.
    
    Parameters:
    - A: numpy array of shape (n, n), a square numeric matrix.
    
    Returns:
    - s: float, the mean of the non-diagonal elements.
    """
    n = A.shape[0]
    
    # Subtract diagonal elements and calculate sum of non-diagonal elements
    non_diag_sum = np.sum(A - np.diag(np.diag(A)))
    
    # Calculate mean of non-diagonal elements
    s = non_diag_sum / (n * (n - 1))
    
    return s

def det_with_lu(K):
    '''
    compute determinant of matrix K with LU decomposition
    '''
    P, L, U = linalg.lu(K)
    det = np.prod(np.diag(U)) * np.linalg.det(P)
    return det


##############################

############################################################
###########  GAMMA APPROXIMATION FUNCTION  #################
############################################################

def get_asymptotic_gamma_params(Kx, Ky):
    '''
    
    '''
    n = Kx.shape[0] 
    ## mean of diag coeffs
    EX = np.mean(np.diag(Kx))
    EY = np.mean(np.diag(Ky))
    
    ## mean of non diag coeffs
    EXX = meannondiag(Kx)
    EYY = meannondiag(Ky)
    
    ## centering gram matrix
    Kx = center_matrix(Kx)
    Ky = center_matrix(Ky)
    
    Bxy = (Kx * Ky)**2
    ## estimation of the mean and variance
    mean = (EX - EXX) * (EY - EYY) /n
    var = 2*(n - 4)*(n -5) / (n* (n - 1) * (n - 2)*(n - 3))* meannondiag(Bxy)
    print('mean asymptotic gamma:', mean, 'var asymptotic gamma:', var)
    
    ## method of moments for Gamma distrib params
    shape = (mean **2) / var
    scale = var / mean
    
    params = {'shape': shape, 'scale': scale}
    
    return params
     
def get_kappa(Kx,Ky):
    """
    ### method taken from causal_learn library on github ###
    Get parameters for the approximated gamma distribution
    Parameters
    ----------
    Kx: kernel matrix for data_x (nxn)
    Ky: kernel matrix for data_y (nxn)

    Returns
    _________
    k_appr, theta_appr: approximated parameters of the gamma distribution

    equivalent to:
        var_appr = 2 * np.trace(Kx.dot(Kx)) * np.trace(Ky.dot(Ky)) / T / T
    based on the fact that:
        np.trace(K.dot(K)) == np.sum(K * K.T), where here K is symmetric
    we can save time on the dot product by only considering the diagonal entries of K.dot(K)
    time complexity is reduced from O(n^3) (matrix dot) to O(n^2) (traverse each element),
    where n is usually big (sample size).
    """
    n = Kx.shape[0]
    Kx = center_matrix(Kx)
    Ky = center_matrix(Ky)
    mean_appr = np.trace(Kx) * np.trace(Ky) / n
    var_appr = 2 * np.sum(Kx ** 2) * np.sum(Ky ** 2) / n / n # same as np.sum(Kx * Kx.T) ..., here Kx is symmetric
    k_appr = mean_appr ** 2 / var_appr
    theta_appr = var_appr / mean_appr
    params = {'shape': k_appr, 'scale': theta_appr }
    return params




#############################################################
######## NON ASYMPTOTIC GAMMA APPROXIMATION FUNCTIONS #######
#############################################################
'''
Method directly inspired by [El Amri and Marrel, 2024]
--> that method is translated from the R package : https://rdrr.io/cran/sensitivity/src/R/testHSIC.R
'''  

 

def gamma_approx( KX, KY):
    ### inputs ###
    # HSIC_obs:     float          (observed value of HSIC index)
    # KX:           n x n          numpy array   (input Gram matrix)
    # KY:           n x n          numpy array   (output Gram matrix)

    ### output ###
    # res:          dict           (containing parameters of Gamma distrib)

    n = KX.shape[0]
    
    # Compute matrices A and W (for HSICXY)
    A = center_matrix(KX)
    W = center_matrix(KY)
    
    # Compute moments for Tr(A W)
    mom = compute_mom_TrAW(A, W)
    
    # Parametric estimation of the p-value
    esp = mom[0]
    var = mom[1]
    # print('normalized mean gamma approx', esp/n**2 )
    # print('normalized var gamma approx', var/n**4)
    # Method of moments for Tr(A W)
    shape_TrAW = (esp**2) / var
    scale_TrAW = var / esp

    # Parameters for HSIC(X, Y) = Tr(A W) / (n^2)
    alpha = shape_TrAW
    beta = scale_TrAW /n**2
    # print('alpha', alpha, 'beta', beta)
    res = {'shape': alpha, 'scale' : beta}
    return res


def compute_mom_TrAW(A, W):
    ### Compute expectations and variances of Tr(A W)
    n = A.shape[0]

    # Denominators for variance formula
    denom1 = ((n-1)**2) * (n+1) * (n-2)
    denom2 = (n+1) * n * (n-1) * (n-2) * (n-3)

    # Matrix operations
    tr_W = np.sum( np.diag(W) )
    tr_W2 = np.sum( np.diag(W) **2)
    sum_W2 = np.sum(W**2)
    # tr_W = np.trace(W)
    # tr_W2 = np.trace(W**2)
    # sum_W2 = np.sum(W*W)

    # Terms used in the final formulas
    O1_W = (n-1) * sum_W2 - tr_W**2
    O2_W = n * (n+1) * tr_W2 - (n-1) * (tr_W**2 + 2 * sum_W2)

    # Matrix A
    tr_A = np.sum( np.diag(W))
    tr_A2 = np.sum(np.diag(A) **2)
    sum_A2 = np.sum(A**2)
    # tr_A = np.trace(A)
    # tr_A2 = np.trace(A**2)
    # sum_A2 = np.sum(A*A)
    # Terms for A
    O1_A = (n-1) * sum_A2 - tr_A**2
    O2_A = n * (n+1) * tr_A2 - (n-1) * (tr_A**2 + 2 * sum_A2)

    # Final formulas for expectation and variance
    esp =( tr_A * tr_W )/ (n-1)
    var = (2 * O1_A * O1_W / denom1 ) + ( O2_A * O2_W / denom2 )

    return np.array([esp, var])



########### Code from Reda El Amri and Gabriel Sarazin to compute the 3 moments of the Pearson 3 distrib #########


def G(A):
    dA = np.diag(A)
    T = np.sum(dA)
    A2c = A**2
    T2 = np.sum(A2c)
    dA2 = dA**2
    S2 = np.sum(dA2)
    A2m = A @ A
    T3 = np.sum(A2m * A)
    S3 = np.sum(dA2 * dA)
    U = np.sum(A2c * A)
    R = np.dot(dA, np.diag(A2m))
    B = np.dot(dA, A @ dA)
    return np.array([T, T2, S2, T3, S3, U, R, B])

def pearson3_moments_cpt(LXi, LY):
    A1 = LXi
    A2 = LY
    A1 = center_matrix(A1)
    A2 = center_matrix(A2)
    n = A1.shape[0]

    # Calculate G for both A1 and A2, and store results in arrays
    out1 = G(A1)
    out2 = G(A2)
    
    # Stack the results from out1 and out2 for easier access
    results = np.vstack([out1, out2])

    T, T2, S2, T3, S3, U, R, B = results[:, 0], results[:, 1], results[:, 2], results[:, 3], results[:, 4], results[:, 5], results[:, 6], results[:, 7]
    print('T pe3', T)
    # Formula for the first moment
    m1_pe3 = np.prod(T) / n + np.prod(-T) / (n * (n - 1))

    # Formula for the second moment
    m2_pe3 = (np.prod(S2) / n +
              (np.prod(T**2 - S2) + 2 * np.prod(T2 - S2) + 4 * np.prod(-S2)) / (n * (n - 1)) +
              (4 * np.prod(2 * S2 - T2) + 2 * np.prod(2 * S2 - T**2)) / (n * (n - 1) * (n - 2)) +
              np.prod(2 * T2 - 6 * S2 + T**2) / (n * (n - 1) * (n - 2) * (n - 3)))

    # Formula for the third moment
    SP1 = np.prod(S3) / n

    SP2 = (4 * np.prod(-S3 + U) + 3 * np.prod(T * S2 - S3) + 6 * np.prod(-S3) +
           12 * np.prod(-S3 + R) + 6 * np.prod(-S3 + B)) / (n * (n - 1))

    SP3 = (3 * np.prod(-T * S2 + 2 * S3) +
           np.prod(T**3 - 3 * T * S2 + 2 * S3) + 12 * np.prod(-T * S2 + 2 * S3 - B) +
           12 * np.prod(2 * S3 - R) + 24 * np.prod(2 * S3 - R - B) +
           6 * np.prod(T * (T2 - S2) + 2 * S3 - 2 * R) +
           24 * np.prod(2 * S3 - U - R) + 8 * np.prod(T3 + 2 * S3 - 3 * R)) / (n * (n - 1) * (n - 2))

    SP4 = (12 * np.prod(T * S2 - 6 * S3 + 2 * R + 2 * B) +
           6 * np.prod(T * (-T2 + S2) - 6 * S3 + 2 * U + 4 * R) +
           3 * np.prod(-T**3 + 5 * T * S2 - 6 * S3 + 2 * B) +
           12 * np.prod(T * (-T2 + 2 * S2) - 6 * S3 + 3 * R + 2 * B) +
           8 * np.prod(-6 * S3 + 2 * U + 3 * R) +
           24 * np.prod(-T3 - 6 * S3 + U + 5 * R + B)) / (n * (n - 1) * (n - 2) * (n - 3))

    SP5 = (3 * np.prod(T**3 + 2 * T * (T2 - 5 * S2) + 24 * S3 - 8 * R - 8 * B) +
           12 * np.prod(T * (T2 - 2 * S2) + 2 * T3 + 24 * S3 - 4 * U - 16 * R - 4 * B)) / (n * (n - 1) * (n - 2) * (n - 3) * (n - 4))

    SP6 = np.prod(-T**3 - 6 * T * (T2 - 3 * S2) - 8 * T3 - 120 * S3 + 16 * U + 72 * R + 24 * B) / (n * (n - 1) * (n - 2) * (n - 3) * (n - 4) * (n - 5))

    m3_pe3 = SP1 + SP2 + SP3 + SP4 + SP5 + SP6

    # Useful characteristics
    mean_pe3 = m1_pe3
    var_pe3 = m2_pe3 - m1_pe3**2
    sk_pe3 = (m3_pe3 - 3 * m1_pe3 * var_pe3 - m1_pe3**3) / (np.sqrt(var_pe3)**3)
    

    return {"mean": mean_pe3, "variance": var_pe3, "skewness": sk_pe3}

def pearson3_param_cpt(LXi, LY):
    mom = pearson3_moments_cpt(LXi, LY)
    n = np.shape(LXi)[0]
    mean_pe3 = mom["mean"]
    var_pe3 = mom["variance"]
    sk_pe3 = mom["skewness"]
    print('for pearson3 approx :', 'mean',mean_pe3/n**2,'var', var_pe3/n**4, 'skew', sk_pe3)
    alpha_pe3 = 4 / (sk_pe3**2)
    beta_pe3 = np.sqrt(var_pe3) * sk_pe3 / 2
    gamma_pe3 = mean_pe3 - 2 * np.sqrt(var_pe3) / sk_pe3

    return {'alpha':alpha_pe3,'beta': beta_pe3, 'gamma': gamma_pe3}



######################################################
########### Random Fourier Features approx ###########
######################################################

def generate_RFF(X, D, sigma):
    """
    Compute an approximate covariance matrix from
    Random Fourier Features (RFF) for a Gaussian (RBF) kernel,
    
    Parameters:
    - X: array of shape (n_samples,d)  (d=1 or d=0 in our setting)
    - D: int, number of random features 
    - sigma: float, the bandwidth parameter of the Gaussian kernel.

    Returns:
    - Gram matrix (n_samples, n_samples) approximated using random Fourier features.
    """

    n = X.shape[0]  # Number of observations
    X = np.reshape(X,(n,1))
    # d = X.shape[1] 
    d = 1
    # Sample random Fourier frequencies W 
    W = np.random.randn(D, d)
    
    # XWt = (1 / np.sqrt(sigma) ) * np.dot(X,W.T) 
    XWt = 1/(sigma**2) * np.dot(X,W.T) 
    # Sample random biases b uniformly from [0, 2*pi]
    # b = np.random.uniform(0, 2 * np.pi, size=D)
    Z1 = np.cos(XWt)
    Z2 = np.sin(XWt)
    
    # Compute random Fourier features Z 
    Z = np.sqrt(1 / D) * np.hstack((Z1,Z2))
    
    return Z

def compute_RFF_hsic(Zx,Zy):
    '''
    - Zx : RFF matrix of size (n x Dx)
    - Zy : RFF matrix of size (n x Dy)
    
    return
    - T : RFF estimator of HSIC 
    '''
    n = np.shape(Zx)[0]
    H = np.identity(n) - 1/n

    ## compute test statistic
    T = 1/(n**2) * np.linalg.norm(Zx.T @ H @ Zy)**2
    
    return T