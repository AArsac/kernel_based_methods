import numpy as np

from scipy import stats,optimize, linalg
from scipy.spatial.distance import cdist, pdist, squareform
# from sklearn.gaussian_process.kernels import RBF

from sklearn.metrics import pairwise_distances
from statsmodels.tsa.stattools import acf


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
def set_width_median(X):
    n = np.shape(X)[0]
    if n > 1000:
        X = X[:1000, :]
    dists = squareform(pdist(X, 'euclidean'))
    width = np.median(dists[dists > 0])
    # width = np.sqrt(2.) * median_dist
    # theta = 1.0 / (width ** 2)
    # width = theta
    return width


def set_bandwidth(data, method):
    '''
    set kernel width parameters in funciton of the method:
    - data = 1d-array
    - method : 'empirical' or 'median'
    '''
    if len(data.shape) == 1:
        data = data.reshape(-1,1)
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


def compute_Gram_matrix(param,X, kernel = 'gaussian',y = None):
    '''
    parameters:
        - param: bandwidth parameter
        - data: data on which the Gram matrix is computed
        - kernel: type of kernel in 'gaussian','laplacian','matern3','matern5'
            where matern3 stands for matern 3/2 and matern 5 stands for matern 5/2
    return:
        K: Gram matrix 
    
    '''
    kwx = param
    
    if len(X.shape) == 1:
        X = X.reshape(-1,1)
    # if len(Y.shape) == 1:
    #     y = y.reshape((y.shape[0],1))
    if y is None:
        y=X
       
    if kernel == 'gaussian':
    ## compute distance in norm2 between each data points
        # distx = cdist(X, y, 'sqeuclidean')
        distx = pairwise_distances(X,y,'sqeuclidean')
        K = np.exp(-0.5 * distx * kwx**-2) #gaussian kernel k(x1,x2) = exp(- 0.5 *||x1-x2||^2 * \sigma_x^-2)
    else:
        ## compute euclidean distance between data points
        # distx = cdist(X,y, metric = 'euclidean')
        distx = pairwise_distances(X,y,'euclidean')
        if kernel == 'laplacian':
            ## compute Laplacian kernel
            K = np.exp(-distx * kwx**-1)
            
        elif kernel =='matern3':
            ## compute Matern 3/2 kernel
            K = distx * np.sqrt(3) * kwx**-1
            K = (1 + K) * np.exp(-K)
            
        elif kernel == 'matern5':
            ## compute Matern 5/2 kernel
            K = distx * np.sqrt(5) * kwx**-1
            K = (1 + K + K**2 / 3) * np.exp(-K)
        else:
            print('kernel not implemented, please select between gaussian, laplacian, matern3, matern5')
            return False
    return K


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



def estimate_head_and_tail(X, Y):
    '''
    Find a time lag A such that the dependence between X_t and Y_t+A 
    is small. The upper bound is the last lag of the series.
    '''
    n = X.shape[0]
    ## Compute auto-correlation of a combined process X,Y
    #  to find index where the dependence becomes weak
    combined = X + Y
    acf_values = acf(combined, nlags=50, fft=True)
    smallest_acf = np.where(acf_values < 0.2)[0] ## first value such that the dependence is under 0.2 (arbitrary)
    if len(smallest_acf) == 0:
        head = 40
    else:
        head = smallest_acf[0]
    if head > min(75, n):
        print('Warning: possibly long memory process, the output of test might be FALSE.')
    head = min(head, 50)
    tail = n
    if (tail - head) < 100:
        print('Warning: using less than 100 points for a bootstrap approximation, stability of the test might be affected')
    
    if (tail-head) > 500:
        tail = 550
    return head, tail

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

# def center_matrix(A):
#     ### Double center the matrix A
#     n, m = A.shape
#     S_mat = np.sum(A) / (n * m)
#     S_rows = np.sum(A, axis=1) / m
#     S_cols = np.sum(A, axis=0) / n

#     B = A - np.outer(S_rows, np.ones(m)) - np.outer(np.ones(n), S_cols) + S_mat
#     return B


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




############################################################
###########  GAMMA APPROXIMATION FUNCTION I.I.D. #################
############################################################

def get_asymptotic_gamma_params(Kx, Ky):
    '''
    asymptotic gamma parameters 
    source: A kernel statistical test of independence, Gretton et al. 2007
    parameters:
        - Kx: double-centered Gram matrix for X
        - Ky: same for Y
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
    # print('mean asymptotic gamma:', mean, 'var asymptotic gamma:', var)
    
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
    # Center matrices --> get A and W 
    A = center_matrix(KX)
    W = center_matrix(KY)
    
    # Compute moments for Tr(A W)
    mom = compute_mom_TrAW(A, W)
    
    # get parameters for 1/n**2 Tr(A,W)
    esp = mom[0]
    var = mom[1]
    # print('normalized mean gamma approx', esp/n**2 )
    # print('normalized var gamma approx', var/n**4)
    
    # Method of moments for 1/n**2 Tr(A W)
    shape_TrAW = (esp**2) / var
    scale_TrAW = var / esp
    
    alpha = shape_TrAW 
    beta = scale_TrAW / n**2
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
    # tr_W = np.sum( np.diag(W) )
    # tr_W2 = np.sum( np.diag(W) **2)
    # sum_W2 = np.sum(W**2)
    tr_W = np.trace(W)
    tr_W2 = np.trace(W**2)
    sum_W2 = np.sum(W*W)

    # Terms used in the final formulas
    O1_W = (n-1) * sum_W2 - tr_W**2
    O2_W = n * (n+1) * tr_W2 - (n-1) * (tr_W**2 + 2 * sum_W2)

    # Matrix A
    # tr_A = np.sum( np.diag(A))
    # tr_A2 = np.sum(np.diag(A) **2)
    # sum_A2 = np.sum(A**2)
    tr_A = np.trace(A)
    tr_A2 = np.trace(A**2)
    sum_A2 = np.sum(A*A)
    # Terms for A
    O1_A = (n-1) * sum_A2 - tr_A**2
    O2_A = n * (n+1) * tr_A2 - (n-1) * (tr_A**2 + 2 * sum_A2)

    # Final formulas for expectation and variance
    esp =( tr_A * tr_W )/ (n-1)
    var = (2 * O1_A * O1_W / denom1 ) + ( O2_A * O2_W / denom2 )

    return np.array([esp, var])


###############################################################
########### Code from Reda El Amri and Gabriel Sarazin ########
######### Compute 3 first moments of Pearson 3 distrib ########
###############################################################

def G(A):
    dA = np.diag(A)
    T = np.sum(dA)
    A2c = A**2
    T2 = np.sum(A2c)
    dA2 = dA**2
    S2 = np.sum(dA2)
    A2m = np.dot(A,A)
    T3 = np.sum(A2m * A)
    S3 = np.sum(dA2 * dA)
    U = np.sum(A2c * A)
    R = np.dot(dA, np.diag(A2m))
    B = np.dot(dA,np.dot(A, dA))
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
    # print('T pe3', T)
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
    sk_pe3 = (m3_pe3 -  3* m1_pe3 * var_pe3 - m1_pe3**3) / (np.sqrt(var_pe3)**3)
    
    return {"mean": mean_pe3, "variance": var_pe3, "skewness": sk_pe3}

def pearson3_param_cpt(LXi, LY):
    mom = pearson3_moments_cpt(LXi, LY)
    n = np.shape(LXi)[0]
    mean_pe3 = mom["mean"]
    var_pe3 = mom["variance"]
    sk_pe3 = mom["skewness"]
    # print('for pearson3 approx :', 'mean',mean_pe3/n**2,'var', var_pe3/n**2, 'skew', sk_pe3)
    alpha_pe3 = 4 / (sk_pe3**2)
    beta_pe3 = np.sqrt(var_pe3) * sk_pe3 / 2
    gamma_pe3 = mean_pe3 - 2 * np.sqrt(var_pe3) / sk_pe3

    return {'alpha':alpha_pe3,'beta': beta_pe3 / n**2, 'gamma': gamma_pe3/ n**2}



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
            /!\ here sigma is 1/(\sigma^2)

    Returns:
    - Gram matrix (n_samples, n_samples) approximated using random Fourier features.
    """

    n = X.shape[0]  # Number of observations
    X = np.reshape(X,(n,1))
    # d = X.shape[1] 
    d = 1
    # Sample random Fourier frequencies W 
    W = np.random.randn(D, d)

    # XWt = sigma * np.dot(X,W.T)
    XWt = sigma * X @ W.T 
    # Sample random biases b uniformly from [0, 2*pi]
    # b = np.random.uniform(0, 2 * np.pi, size=D)
    Z1 = np.cos(XWt)
    Z2 = np.sin(XWt)
    
    # Compute random Fourier features Z 
    # Z = np.sqrt(1 / D) * np.hstack((Z1,Z2))
    Z = np.sqrt(1/D) * np.hstack((Z1,Z2))
    # print('shape Z = ', np.shape(Z))
    return Z

def compute_RFF_hsic(Zx,Zy):
    '''
    - Zx : RFF matrix of size (n x Dx)
    - Zy : RFF matrix of size (n x Dy)
    
    return
    - stat : RFF estimator of HSIC 
    '''
    n = Zx.shape[0]
    # H = np.eye(n) - np.ones((n, n)) / n  # Centering matrix
    ## center feature matrices
    # Zx_c = H @ Zx                        
    Zy_c = Zy - np.mean(Zy, axis=0, keepdims=True)  # Efficient H @ Zy
    
    product = Zx.T @ Zy_c             # (2D x 2D) matrix
    
    stat = np.sum(product**2) / (n**2)  # Frobenius norm squared / n^2
    
    return stat



def rff_center_matrix(Z):
    """ Efficient centering of matrix Z, without explicitly forming centering matrix """
    n = Z.shape[0]  # Number of samples (rows)
    
    # Compute row and column means efficiently
    row_means = np.mean(Z, axis=0)  # Shape (1, D)
    col_means = np.mean(Z, axis=1)  # Shape (n,)
    
    # Subtract row means from each row and column means from each column
    Z_centered = Z - row_means
    Z_centered = Z_centered - col_means[:, np.newaxis]
    
    return Z_centered


def RFF_gamma_params(Zx, Zy):
    '''
    Compute the gamma parameters using the approximation of the Gram matrices
    \Lx \approx \Zx \Zx^T and $\Ly \approx \Zy \Zy^T
    '''
    n = np.shape(Zx)[0]

    # Center the feature matrices directly
    Zx_c = Zx - np.mean(Zx, axis=0, keepdims=True)  # shape (n x D)
    Zy_c = Zy - np.mean(Zy, axis=0, keepdims=True)  # shape (n x D)

    # Compute feature covariances (D x D)
    tZx = Zx_c.T @ Zx_c
    tZy = Zy_c.T @ Zy_c
    # Compute the expectations (traces)
    
    tr_tZx = np.trace(tZx)  # Trace of tZx
    tr_tZy = np.trace(tZy)  # Trace of tZy
    
    # Compute the squared traces

    tr_tZx_squared = np.sum(tZx**2)  # Trace of tZx^2
    tr_tZy_squared = np.sum(tZy**2)  # Trace of tZy^2
    
    # Compute the expectation and variance
    n = Zx.shape[0]  # Number of samples 
    E_Sn = (1 / n**2) * tr_tZx * tr_tZy  # Expectation of S_n
    Var_Sn = (2 / n**4) * tr_tZx_squared * tr_tZy_squared  # Variance of S_n
    
    # Compute Gamma parameters
    beta = (E_Sn ** 2) / Var_Sn
    gamma = Var_Sn / (n * E_Sn)
    
    return {'shape': beta, 'scale': gamma}

# def RFF_gamma_params(Zx, Zy):
#     '''
#     Compute parameters of the Gamma approximation
#     Consider the Covariance matrices $\Zx^T\Zx$ and $\Zy^T \Zy$
#     to replace the Gram matrices 
    
#     '''
#     # Center the feature matrices
#     mean_Zx = np.mean(Zx, axis=0)
#     mean_Zy = np.mean(Zy, axis=0)

#     centered_Zx = Zx - mean_Zx
#     centered_Zy = Zy - mean_Zy

#     # Compute the centered covariance matrices
#     Cx = centered_Zx.T @ centered_Zx  # size (2D x 2D)
#     Cy = centered_Zy.T @ centered_Zy  # size (2D x 2D)

#     # Compute the expectations (traces)
#     tr_Cx = np.trace(Cx)  # Trace of Cx
#     tr_Cy = np.trace(Cy)  # Trace of Cy

#     # Compute the squared traces
#     tr_Cx_squared = np.trace(Cx @ Cx)  # Trace of Cx^2
#     tr_Cy_squared = np.trace(Cy @ Cy)  # Trace of Cy^2

#     # Compute the expectation and variance
#     n = Zx.shape[0]  # Number of samples
#     E_Sn = (1 / n**2) * tr_Cx * tr_Cy  # Expectation of S_n
#     Var_Sn = (2 / n**4) * tr_Cx_squared * tr_Cy_squared  # Variance of S_n

#     # Compute Gamma parameters
#     beta = (E_Sn ** 2) / Var_Sn
#     gamma = Var_Sn / (n * E_Sn)

#     return {'shape': beta, 'scale': gamma}

# def RFF_gamma_params(Zx,Zy):

#     '''
#     n: sample size
#     Zx: such that Lx \approx Zx^T Zx
#     '''
#     ## center covariance matrices
#     n = np.shape(Zx)[0]
#     # Zx = Zx - np.mean(Zx, axis=0, keepdims=True)
#     tZx = center_rff_matrix(Zx)@ center_rff_matrix(Zx).T
#     tZy = center_rff_matrix(Zy) @ center_rff_matrix(Zy).T
#     # Zy = Zy - np.mean(Zy, axis=0, keepdims=True)
#     print('shape Zx: ',np.shape(tZx))
    
#     # mean_appr = np.trace(Zx) * np.trace(Zy) / n
#     # var_appr = 2 * np.sum(Zx ** 2) * np.sum(Zy ** 2) / n**2 # same as np.sum(Kx * Kx.T) ..., here Kx is symmetric
    
#     # k_appr = mean_appr ** 2 / var_appr
#     # theta_appr = var_appr / mean_appr
#     # params = {'shape': k_appr, 'scale': theta_appr }
#     # return params
    
#         # Compute trace-based estimates
#     trace_tZx = np.trace(tZx)
#     trace_tZy = np.trace(tZy)
#     trace_tZx2 = np.trace(tZx @ tZx)
#     trace_tZy2 = np.trace(tZy @ tZy)

#     # Compute expectation and variance
#     expectation = (1 / n) * trace_tZx * trace_tZy
#     variance = (2 / n**2) * trace_tZx2 * trace_tZy2
#         # Compute gamma parameters
#     beta = expectation**2 / variance
#     gamma_scale = variance / (n* expectation)  # Scale parameter for scipy's gamma
    
#     return {'shape': beta, 'scale': gamma_scale}
