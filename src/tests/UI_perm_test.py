import numpy as np
from numpy.linalg import eigh, eigvalsh
from scipy import stats,optimize
from scipy.spatial.distance import cdist
# from sklearn.gaussian_process.kernels import RBF


from itertools import permutations, combinations
import math


import src.tests.utils as utils


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




# def permutation_test(data_x,data_y, param = [], method = 'median',  nb_permut = 500):
#     '''
    
    
#     parameters :
#     - data_x: 1-D array (n x 1)
#     - data_y: 1-D array (n x 1)
#     - param: np.array([\sigma_x, \sigma_y]) bandwidths parameters for kernel x, kernel y
#     - method: 'median' or 'empirical'
#     - nb_permu: integer, number of permutations for the test
    
#     returns:
#         dict : {'pval' : pval, 'test_statistic': T}
#     '''
#     # data_x = stats.zscore(data_x, ddof=1, axis=0)
    
#     # data_y = stats.zscore(data_y, ddof=1, axis=0)
#     if len(param)==0:
#         sigma_x = utils.set_bandwidth(data_x, method = method)
#         sigma_y = utils.set_bandwidth(data_y, method = method)
#         param = np.array([sigma_x,sigma_y])
#     Lx = utils.compute_Gram_matrix(param[0], data_x)
#     Ly = utils.compute_Gram_matrix(param[1], data_y)
#     T = compute_V_stat(Lx,Ly)

#     # Ly_permuted_list = generate_permuted_matrices(Ly,nb_permut)
#     Y_perm = utils.generate_unique_permutations(data_y,nb_permut)
#     T_perm_list = []
#     for j in range(nb_permut):
#         Ly_perm = utils.compute_Gram_matrix(param[1],Y_perm[j])
#         T_perm = compute_V_stat(Lx,Ly_perm)
#         T_perm_list.append(T_perm)
        
#     #threshold = np.quantile(T_perm_list, 1-alpha)
#     #reject = T > threshold

#     pval = (T < T_perm_list).mean()
    
#     results = {'pval': pval, 'test_statistic': T}
#     return results

def permutation_test(data_x,data_y, param = [], method = 'median',  nb_permut = 500):
    '''
    
    
    parameters :
    - data_x: 1-D array (n x 1)
    - data_y: 1-D array (n x 1)
    - param: np.array([\sigma_x, \sigma_y]) bandwidths parameters for kernel x, kernel y
    - method: 'median' or 'empirical'
    - nb_permu: integer, number of permutations for the test
    
    returns:
        dict : {'pval' : pval, 'test_statistic': T}
    '''
    # data_x = stats.zscore(data_x, ddof=1, axis=0)
    
    # data_y = stats.zscore(data_y, ddof=1, axis=0)
    
    if len(param)==0:
        sigma_x = utils.set_bandwidth(data_x, method = method)
        sigma_y = utils.set_bandwidth(data_y, method = method)
        param = np.array([sigma_x,sigma_y])
    Lx = utils.compute_Gram_matrix(param[0], data_x)
    Ly = utils.compute_Gram_matrix(param[1], data_y)
    T = compute_V_stat(Lx,Ly)
    n = Ly.shape[0]
    T_perm_list = []
    for _ in range(nb_permut):
        shuf = np.random.permutation(n)
        # Shuffle Ly by permuting rows and columns
        Ly_perm = Ly[np.ix_(shuf, shuf)]
        T_perm = compute_V_stat(Lx,Ly_perm)
        T_perm_list.append(T_perm)
        
    #threshold = np.quantile(T_perm_list, 1-alpha)
    #reject = T > threshold

    pval = (T < T_perm_list).mean()
    
    results = {'pval': pval, 'test_statistic': T}
    return results




if __name__ == '__main__':
    import timeit
    #l = np.random.randint(low = 1, high = 11)  # Choose any value from 1 to 10
    l = 3
    n = 700

    # Generate L, Theta, epsilon_1, epsilon_2
    L = np.random.choice(np.arange(1, l + 1), n)
    Theta = np.random.uniform(0, 2 * np.pi, n)
    epsilon_1 = np.random.normal(0, 1, n)
    epsilon_2 = np.random.normal(0, 1, n)

    # Simulate X and Y
    X = L * np.cos(Theta) + epsilon_1 / 4
    Y = L * np.sin(Theta) + epsilon_2 / 4

    ## TEST

    B = 500 # number of permutations
    start = timeit.default_timer()
    res1 = permutation_test(X,Y,method = 'median',nb_permut = B)
    stop = timeit.default_timer()
    print(res1['pval'], res1['test_statistic'], 'time :', stop - start)
    
    


