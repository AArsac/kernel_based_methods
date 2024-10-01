'''
Method directly inspired by [El Amri and Marrel, 2024]
--> that method is translated from the R package : https://rdrr.io/cran/sensitivity/src/R/testHSIC.R

'''

# import src.utils as utils
import src.tests.utils as utils
import numpy as np
from scipy.stats import gamma



def compute_V_stat(Lx,Ly):
    '''
    Compute the estimator of HSIC : T = 1/n**2 Tr(L_xL_y),
    with L_x, L_y doubly centered Gram matrices
    
    Do not compute Tr(L_x L_y) directly, use the fact that 
    Tr(L_x L_y) = Sum(L_x * L_y) where A*B is the Hadamard between A and B
    '''
    n = Lx.shape[0]

    ##doubly center the kernel matrices
    Lx = utils.center_matrix(Lx)
    Ly = utils.center_matrix(Ly)
    # compute the test statistic
    T = 1/n**2 * np.sum(Lx * Ly)
    return T


def gamma_test(data_x, data_y, param = [], method = 'median'):
    '''
    
    '''
    if len(param)==0:
        sigma_x = utils.set_bandwidth(data_x, method = method)
        sigma_y = utils.set_bandwidth(data_y, method = method)
        param = np.array([sigma_x,sigma_y])
    Lx = utils.compute_Gram_matrix(param[0], data_x)
    Ly = utils.compute_Gram_matrix(param[1], data_y)
    T = compute_V_stat(Lx,Ly)
    
    pval = gamma_approx(T, Lx, Ly)
    
    results = {'pval': pval, 'test_statistic': T}
    return results


def gamma_approx(HSIC_obs, KX, KY):
    ### inputs ###
    # HSIC_obs:     float          (observed value of HSIC index)
    # KX:           n x n          numpy array   (input Gram matrix)
    # KY:           n x n          numpy array   (output Gram matrix)

    ### output ###
    # res:          dict           (test result, containing 'pval' and 'param')

    n = KX.shape[0]
    
    # Compute matrices A and W (for HSICXY)
    A = utils.center_matrix(KX)
    W = utils.center_matrix(KY)
    
    # Compute moments for Tr(A W)
    mom = compute_mom_TrAW(A, W)
    
    # Parametric estimation of the p-value
    esp = mom[0]
    var = mom[1]

    # Method of moments for Tr(A W)
    shape_TrAW = (esp**2) / var
    scale_TrAW = var / esp

    # Parameters for HSIC(X, Y) = Tr(A W) / (n^2)
    alpha = shape_TrAW
    beta = scale_TrAW / (n**2)

    # Parametric estimation of the p-value
    pval = 1 - gamma.cdf(HSIC_obs, a=alpha, scale=beta)
    
    return pval


def compute_mom_TrAW(A, W):
    ### Compute expectations and variances of Tr(A W)
    n = A.shape[0]

    # Denominators for variance formula
    denom1 = ((n-1)**2) * (n+1) * (n-2)
    denom2 = (n+1) * n * (n-1) * (n-2) * (n-3)

    # Matrix operations
    tr_W = np.trace(W)
    tr_W2 = np.trace(W @ W)
    sum_W2 = np.sum(W**2)

    # Terms used in the final formulas
    O1_W = (n-1) * sum_W2 - tr_W**2
    O2_W = n * (n+1) * tr_W2 - (n-1) * (tr_W**2 + 2 * sum_W2)

    # Matrix A
    tr_A = np.trace(A)
    tr_A2 = np.trace(A @ A)
    sum_A2 = np.sum(A**2)

    # Terms for A
    O1_A = (n-1) * sum_A2 - tr_A**2
    O2_A = n * (n+1) * tr_A2 - (n-1) * (tr_A**2 + 2 * sum_A2)

    # Final formulas for expectation and variance
    esp = tr_A * tr_W / (n-1)
    var = 2 * O1_A * O1_W / denom1 + O2_A * O2_W / denom2

    return np.array([esp, var])





if __name__ == '__main__' : 
    from UI_perm_test import permutation_test
    
    #####################
    ##  SIMULATE DATA  ##
    #####################
    l = 1
    n = 300
    
    X = np.zeros((n,))
    Y = np.zeros((n,))
    for j in range(n):
        L = np.random.choice(np.arange(1, l + 1))
        Theta = np.random.uniform(0, 2 * np.pi)
        epsilon_1 = np.random.normal(0, 1)
        epsilon_2 = np.random.normal(0, 1)

        # Simulate X and Y
        X[j] = L * np.cos(Theta) + epsilon_1 / 4
        Y[j] = L * np.sin(Theta) + epsilon_2 / 4
    
    
    res = gamma_test(X, Y)
    
    print('pval = ', res['pval'] ,' and test statistic value =', res['test_statistic'] )
    
    res_perm = permutation_test(X,Y)
    
    print('pval = ', res_perm['pval'] ,' and test statistic value =', res_perm['test_statistic'] )