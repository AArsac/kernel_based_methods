'''
Objective here is to approx the spectral law given in Chwialkowski & Gretton 2014
'''

import numpy as np

from scipy import stats,optimize
from scipy.spatial.distance import cdist
# from sklearn.gaussian_process.kernels import RBF

from statsmodels.tsa.stattools import acf


from src.tests import utils
# from numpy.linalg import eigh, eigvalsh
from scipy.sparse.linalg import eigsh

from scipy.stats import gamma, pearson3


def compute_V_stat(Lx,Ly):
    '''
    Compute the estimator of HSIC : T = 1/n**2 Tr(L_xL_y),
    with L_x, L_y doubly centered Gram matrices with size (n x n)
    
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

def estimate_eigvalues(K, nb_eig):
    '''
    estimate eigenvalues and eigenvectors of the matrix K
    ordered in descending order 
    nb_eig : int, indicate the number of eigenvalues to keep
    '''
    n = np.shape(K)[0]
    mu_x, v_x = eigsh(K, k= nb_eig, which = 'LM') 

    ## Sort eigenvalues in descending order and reorder eigenvectors accordingly
    sorted_indices = np.argsort(mu_x)[::-1]  
    mu_x = mu_x[sorted_indices]  
    v_x = v_x[:, sorted_indices]  
    
    ## normalize eigenvectors and eigenvalues
    v_x = np.sqrt(n) * v_x
    mu_x = 1/n * mu_x
    return mu_x, v_x


# def eigenfunction(K, v, lbd):
#     '''
#     Compute eigenfunctions from Nyström method 
#     (--> see Kernel PCA )
    
#     K: Gram matrix
#     v: eigenfunctions of K
#     lbd: corresponding eigenvalues
#     '''
#     n = np.shape(K)[0]
#     r = len(lbd)
#     phi = np.zeros((n,r))
#     for i, lbd_i in enumerate(lbd):
#         phi[:,i] = 1/np.sqrt(lbd_i) *np.dot(K, v[:,i])
#     print('shape phi',phi.shape)
#     return phi

     
def covariance_matrix(V_x,V_y, As):
    '''
    For data x = (x_1,...,x_n),y = (y_1,...,y_n) generated from weakly dependent stationary random processes:
    
    compute matrix \Sigma with 
    \Sigma_{(i,j),(k,l)} = E[u_{X,i}(X_0)u_{X,k}(X_0)u_{Y,j}(Y_0)u_{Y,l}(Y_0)] 
                        + \sum_{t=1}^As E[u_{X,i}(X_0)u_{X,k}(X_{t+1})u_{Y,j}(Y_0)u_{Y,l}(Y_{t+1})]
                        + \sum_{t=1}^As E[u_{X,i}(X_{t+1})u_{X,k}(X_0)u_{Y,j}(Y_{t+1})u_{Y,l}(Y_0)]
                        = A_{ijkl} + \sum_{t=1}^As B_{t,ijkl} + \sum_{t=1}^As C_{t,ijkl}
    
    Note here that As is the first step k \in \mathbb{N} such that Cov(X_t, Y_t+k) \leq \epsilon, for \epsilon > 0 
    
    parameters:
        - V_x: matrix of (estimated) eigenfunctions of k_x
        - V_y: matrix of (estimated) eigenfunctions of k_y
        - As: minimum lag 
    return:
        - Sigma: covariance matrix of size (r^2 x r^2), with r number of eigenvectors 
    '''
    r = np.shape(V_x)[1]
    
    
    Sigma = np.zeros((r**2,r**2))

    ## Compute Sigma
    ## As Sigma is a covariance matrix, only triangular upper (or lower) terms need to be calculated
    
    def inner_product(func1, func2, start=0, lag=0):
        # Compute estimation of E[u_i(x_t)*u_k(x_{t+1})]
        # using empirical mean to estimate expectation
        return np.mean(func1[start:-1-lag] * func2[start+lag:-1])   ## adjust size of one vect w.r.t. to the other

    for i in range(r):
        for j in range(r):
            for k in range(i,r): ## Only upper triangle
                for l in range(j  if k == i else 0, r):  ## Ensure symmetry within blocks
                    # for matrix indices
                    idx1 = i * r + j
                    idx2 = k * r + l
                    
                    # Compute A_{ijkl}
                    A_ijkl = (
                        inner_product(V_x[:,i], V_x[:,k]) 
                        * inner_product(V_y[:,j], V_y[:,l])
                    )
                    
                    # Compute the sum over B_t and C_t
                    sum = 0
                    for t in range(1, As):
                        B_t = (
                            inner_product(V_x[:,i], V_x[:,k], start = 0, lag = t)
                            * inner_product(V_y[:,j], V_y[:,l], start = 0, lag = t)
                        )
                        C_t = (
                            inner_product(V_x[:,k], V_x[:,i], start = 0, lag = t)
                            * inner_product(V_y[:,l], V_y[:,j], start = 0, lag = t)
                        )
                        sum += B_t+C_t
                    # compute Sigma_{(i,j),(k,l)}
                    Sigma[idx1, idx2] = A_ijkl + sum
                    
                    ## fill the symmetric part
                    if idx1 != idx2:
                        Sigma[idx2,idx1] = Sigma[idx1,idx2]

    return Sigma


def get_asymptotic_sum(mu_x, mu_y, Sigma):
    '''
    Compute asymptotic dsitribution of the V-stat estimator of the HSIC
    using Chwialkowski & Gretton 2014 result:
    nH_v(X,Y) converges in distribution to \sum_i\sum_j \lambda_i \lambda_j z_{ij}^2
    
    parameters:
        - mu_x: (estimated) eigenvalues of integral operator of kernel k_x
        - mu_y: (estimated) eigenvalues of integral operator of kernel k_y
        - Sigma: Covariance matrix as defined by Chwialkowski and Gretton 2014
    returns:
        - spectral_var: a simulated variable following the spectral distribution 
    '''
    r = len(mu_x)
    
    ##  \mu_z = \mu_x \mu_y
    mu_z = np.outer(mu_x,mu_y).flatten()
    ## compute covariance matrix
    
    ## draw Z following N(0, Sigma)
    Z = np.random.multivariate_normal(mean = np.zeros(r**2), cov = Sigma)
    
    ## compute the sum 
    ## /!\ here the sum is not normalized by the sample_size !!
    spectral_var =  sum(mu_z*Z**2)

    return spectral_var



                
                    

def spectral_hsic_test(data_x, data_y, param = [], method = 'median', head = 10, approx_min_lag = 'estimated',kernel = 'gaussian', nb_eigvals = 5, nb_samples = 500 ):
    '''
    Compute the HSIC test of independence using its spectral distribution to simulate
    the null distribution $X_t \perp Y_t$, as given by Chwialkowski & Gretton 2014
    
    parameters:
        - data_x, data_y: time series data X_t and Y_t of size (n x 1) resp.
        - param: fixed kernel bandwidths parameters
        - method: 'median' or 'empirical' if no bandwidths parameters are given, define them automatically
        - head: fixed A parameter in the shifting procedure --> At which lag to start the shifts (such that the autocorrelation is weak enough)
        - tail: fixed B param in the shifting procedure 
        - approx_min_lag: 'empirical' or 'estimated' --> To estimate parameters head and tail. 
                        If 'estimated', use acf to determine which lag in the series is independent enough of a fixed time step t
        - kernel: which kernel to use: 'gaussian', 'matern3', 'matern5', 'laplacian'
        - nb_eigvals: number of eigevalues to compute
        - nb_samples: number of samples to simulate the null distribution
    returns:
    - results: dict{'pval', 'test_statistic', 'test_stat_H0'}
    '''
    if len(param) == 0:
        print('bandwidth parameters set with median heuristic')
        sigma_x = utils.set_bandwidth(data_x, method=method)
        sigma_y = utils.set_bandwidth(data_y, method=method)
        param = np.array([sigma_x, sigma_y])
        
    ## Compute the autocorrelation function
    
    if approx_min_lag == 'empirical':
        As = head

    elif approx_min_lag == 'estimated':
        As,B = utils.estimate_head_and_tail(data_x, data_y)
        if As > 20:
            As = 20
    print('minimal lag such that autocorrelation is low As = ',As)
    ### Compute Gram matrices of kx and ky from X and Y
    Lx = utils.compute_Gram_matrix(param[0], data_x, kernel = kernel)
    Ly = utils.compute_Gram_matrix(param[1], data_y, kernel = kernel)

    ## Center Gram matrices
    Lx_centered = utils.center_matrix(Lx)
    Ly_centered = utils.center_matrix(Ly)
    
    ## Estimate the HSIC using its V-statistic estimator
    T = compute_V_stat(Lx, Ly)
    
    ####   ESTIMATE THE NULL DISTRIBUTION   ####
    
    ## Estimate eigenvalues
    mu_x, V_x = estimate_eigvalues(Lx_centered, nb_eigvals)
    mu_y, V_y = estimate_eigvalues(Ly_centered, nb_eigvals)
    
    ## compute covariance matrix
    Sigma = covariance_matrix(V_x, V_y, As)

    ## Estimate the spectral distribution
    spectral_distrib = [1/len(data_x) * get_asymptotic_sum(mu_x, mu_y, Sigma) for _ in range(nb_samples) ]
    
    ## Compute the p-value
    
    pval = (T < spectral_distrib).mean()
    results = {'pval':pval, 'test_statistic': T, 'test_stat_H0': spectral_distrib}
    
    
    return results





###################################################
##### ASYMPTOTIC GAMMA APPROXIMATION NON-IID  #####
###################################################

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

def compute_expectation(lbd_x,lbd_y,Sigma):
    '''
    Compute the expectation of the spectral approximation
    of the HSIC for weakly dependent data as defined in
    Chwialkowski and Gretton 2014
    The expectation is given by:
        \sum_p \sum_q \lambda_{X,p} \lambda_{X,q} *Sigma_{pq, pq}
    Where Sigma is the covariance matrix defined in the same article
    
    parameters:
        - lbd_x: list of (normalized) eigenvalues of the kernel K_x
        - lbd_y: list of (normalized) eigenvalues of the kernel K_y
        - Sigma: covariance matrix of size (r^2 \times r^2)
    returns:
        - Expectation 
    '''

    ## compute the product of eigenvalues
    lbd_z=  np.outer(lbd_x,lbd_y).flatten()
    
    ## extract diagonal terms of the covariance matrix
    diag_sigma = np.diag(Sigma)
    
    ## compute the sum
    exp = np.sum(lbd_z * diag_sigma)

    return exp

def compute_variance(lbd_x, lbd_y, Sigma):
    """
    Compute the variance of the spectral distribution
    
    Var(S)  = 2*sum_{p,q,r,s} \lambda_X,p  \lambda_Y,q  \lambda_X,r \lambda λ_Y,s * (Sigma_{(pq),(rs)})^2
    
    Parameters:
        - lbd_x,lbd_y: list of eigenvalues of kernel k_x/k_y
        - Sigma: Covariance matrix of shape (r^2, r^2).
    
    Returns:
        - variance
    """
    # Compute the outer product of eigenvalues + resize it
    lbd_z = np.outer(lbd_x, lbd_y).flatten()

    # Compute the variance 
    variance =  2*np.sum(Sigma**2 * np.outer(lbd_z, lbd_z))

    return variance


def gamma_parameters(n, lbd_x, lbd_y, Sigma):
    '''
    Compute the parameters of the gamma distribution
    /!\ Our test statistic is 1/n^2 Tr(Lx Ly) but the
    converge to the spectral distribution is shown for 1/n Tr(Lx Ly) !
    Those it is necessary to multiply by (1/n) the expectation
    and (1/n^2) the variance obtained from the spectral distribution
    
    parameters:
        - n: int, sample size
        - lbdx, lbd_z: lists of size r, eigenvalue of the double-centered kernel matrices
        - Sigma: numpy array of size r^2 \times r^2, Covariance matrix 
    returns:
        - parameters of the gamma distribution
    
    '''

    exp = compute_expectation(lbd_x, lbd_y, Sigma) / n
    # exp = compute_expectation(Lx, Ly, Sigma) / n
    var = compute_variance(lbd_x, lbd_y, Sigma) / n**2
    k_appr = exp ** 2 / var 
    
    theta_appr = var / exp
    params = {'shape': k_appr, 'scale': theta_appr }
    return params


def gamma_spectral_test(data_x, data_y, param = [], method = 'median', head = 10, approx_min_lag = 'estimated', kernel = 'gaussian', nb_eigvals = 5):
    '''
    Compute the HSIC test of independence using a Gamma approximation
    to simulate the null distribution $X_t \perp Y_t$
    
    parameters:
        - data_x, data_y: time series data X_t and Y_t of size (n x 1) resp.
        - param: fixed kernel bandwidths parameters
        - method: 'median' or 'empirical' if no bandwidths parameters are given, define them automatically
        - head: fixed A parameter in the shifting procedure --> At which lag to start the shifts (such that the autocorrelation is weak enough)
        - tail: fixed B param in the shifting procedure 
        - approx_min_lag: 'empirical' or 'estimated' --> To estimate parameters head and tail. 
                        If 'estimated', use acf to determine which lag in the series is independent enough of a fixed time step t
        - kernel: which kernel to use: 'gaussian', 'matern3', 'matern5', 'laplacian' 
                (Matern3 stands for Matern 3/2, same or matern5)
        - nb_eigvals: number of eigevalues to compute
        
    returns:
    - results: dict{'pval', 'test_statistic', 'test_stat_H0'}
    '''
    if len(param) == 0:
        print('bandwidth parameters set with median heuristic')
        sigma_x = utils.set_bandwidth(data_x, method=method)
        sigma_y = utils.set_bandwidth(data_y, method=method)
        param = np.array([sigma_x, sigma_y])
        
    ## Compute the autocorrelation function
    
    if approx_min_lag == 'empirical':
        As = head

    elif approx_min_lag == 'estimated':
        As,B = utils.estimate_head_and_tail(data_x, data_y)
        if As > 20:
            As = 20
    ### Compute Gram matrices of kx and ky from X and Y
    Lx = utils.compute_Gram_matrix(param[0], data_x, kernel = kernel)
    Ly = utils.compute_Gram_matrix(param[1], data_y, kernel = kernel)

    ## Center Gram matrices
    Lx_centered = utils.center_matrix(Lx)
    Ly_centered = utils.center_matrix(Ly)
    
    ## Estimate the HSIC using its V-statistic estimator
    T = compute_V_stat(Lx, Ly)
    
    ####   ESTIMATE THE NULL DISTRIBUTION   ####
    
    ## Estimate eigenvalues
    mu_x, V_x = estimate_eigvalues(Lx_centered, nb_eigvals)
    mu_y, V_y = estimate_eigvalues(Ly_centered, nb_eigvals)
    
    ## compute covariance matrix
    Sigma = covariance_matrix(V_x, V_y, As)
    
    ## get the gamma parameters
    n = len(data_x)
    gamma_params = gamma_parameters(n,mu_x, mu_y, Sigma)
    
    pval  = 1 - gamma.cdf(T, a = gamma_params['shape'], scale = gamma_params['scale'])

    results = {'pval': pval, 'test_statistic': T, 'params': gamma_params }
    
    return results






    
if __name__ == '__main__':

    import os
    import sys
    import pickle
    import pandas as pd
    import matplotlib.pyplot as plt
    from src.tests.utils import set_bandwidth, compute_Gram_matrix, center_matrix
    from src.tests.UI_perm_test import permutation_test, asymptotic_test
    from src.tests.shift_hsic import shift_test, RFF_shift_test
    # from src.ts_hsic_asympt.RBF_asympt import get_asymptotic_distrib

    # current_dir = os.path.dirname(os.path.abspath(__file__))

    # main_dir = os.path.dirname(os.path.dirname(current_dir) )
    
    # sample_size = 500
    
    
    # paramX = 0.2
    # paramY = 0.22
        
    # alpha = 0.2
    # beta = 0.2
   
    # X_t = np.random.uniform(0,0.5, size = (sample_size, ))
    # Y_t = np.random.normal(size = (sample_size, ))

    # for j in range(1,sample_size):
    #     X_t[j] = alpha * X_t[j-1] + np.random.normal(scale = paramX)
    #     Y_t[j] = beta * Y_t[j-1] +  np.random.normal(scale = paramY)

    # data =  {'X':X_t,'Y':Y_t}
    
    # X = data['X']
    # Y = data['Y']
    # # kernels = ['gaussian', 'laplacian', 'matern3','matern5']
    # kernels = ['gaussian']
    # r = 5
    # plt.figure(figsize=(12,6))
    # for k in kernels:
    #     spectral_T = spectral_hsic_test(X, Y, param = np.array([0.5, 0.5]), nb_eigvals= r, kernel = k )
    #     spectral_H0 = spectral_T['test_stat_H0']
        
    #     nb_samples = 1000
    #     ###### GAMMA APPROX IID ######
    #     T_gamma_iid = asymptotic_test(X, Y, param = np.array([0.5, 0.5]))
    #     gamma_param_iid = T_gamma_iid['params']
    #     xx= np.linspace(0,np.max(spectral_H0)+spectral_T['test_statistic'],nb_samples)
                
    #     pdf_gamma_iid = gamma.pdf(xx, a = gamma_param_iid['shape'], scale = gamma_param_iid['scale'])
        
    #     # #### GAMMA APPROXIMATION NON IID
    #     T_gamma = gamma_spectral_test(X, Y, param = np.array([.5,.5]), nb_eigvals= 10 )
        
    #     large_gamma_param = T_gamma['params']
        
        
    #     pdf_large_gamma = gamma.pdf(xx, a = large_gamma_param['shape'], scale = large_gamma_param['scale'])
        
        
                
    #     plt.plot(xx, pdf_large_gamma,linestyle = 'dashed',color = 'skyblue', linewidth = 2, label = 'Gamma-Spectral-HSIC')
    #     plt.plot(xx, pdf_gamma_iid, linestyle = 'dashed', linewidth = 2,color = 'red', label= 'Gamma-HSIC')
        
        
        
    #     plt.hist(spectral_H0, density = True, alpha = 0.6,color = 'skyblue', label = f'Spectral-HSIC' )
        
        
    #     shift_t = shift_test(X,Y, param = np.array([0.5, 0.5]), approx_min_lag= 'estimated', kernel = k, nb_shifts = 500)
    #     T_shift_H0 = shift_t['test_stat_H0']
        
    #     plt.hist(T_shift_H0,density = True, alpha = 0.6, color = 'salmon',label = f'Shift-HSIC')
    
    # plt.xlabel('HSIC values', fontsize =20)
    # plt.ylabel('PDF values', fontsize = 20)
    # plt.legend(fontsize = 25)
    # save_fig_path = main_dir + f'/plot/spectral_distrib/ts/n{sample_size}'
    # plt.savefig(save_fig_path + f'/True_spectral_distrib_n{sample_size}_r{r}_alpha{alpha}_kernel_{k}.pdf', format = 'pdf')
    # plt.show()

########################################################################################
########################################################################################
    
    ### GENERATE DATA #### 
    ### X_t and Y_t are AR(1) processes
    sample_size = 1000
    nn = [200,300,500,1000]
    # nn = [1000]
    for sample_size in nn:
    #     save_fig_path = main_dir+f'/plot/spectral_distrib/ts/n{sample_size}'
        
    #     ## standard deviation for the additional noise in X_t and Y_t
    #     ## Or standard deviation of the normal distribution in iid setting
        paramX = 0.22
        paramY = 0.2

    #     ## ts: wether data are time series or not
    #     ts = False

    #     if ts:
    #         ## auto correlation coeffs for X_t and Y_t
        alpha = 0.2
        beta = 0.2

        
        X_t = np.random.uniform(0,0.5, size = (sample_size, ))
        Y_t = np.random.normal(size = (sample_size, ))

        for j in range(1,sample_size):
            X_t[j] = alpha * X_t[j-1] + np.random.normal(scale = paramX)
            Y_t[j] = beta * Y_t[j-1] +  np.abs(X_t[j-1]) +  np.random.normal(scale = paramY)

        data =  {'X':X_t,'Y':Y_t}


    #     else:
    #         alpha = 0
    #         beta = 0
    #         data = {'X':np.random.normal(scale = np.sqrt(paramX),size = (sample_size, )), 'Y': np.random.normal(scale = np.sqrt(paramY), size = (sample_size, ))}
    #     X = data['X']
    #     Y = data['Y']
        
        sig_x = np.sqrt( 0.2 )
        sig_y = np.sqrt( 0.2 )
    #     if ts == False:
    #         paramX = np.var(X)
    #         paramY = np.var(Y)
            
    #     params = {'alpha':alpha, 'beta':beta, 
    #             'var_x':paramX, 'var_y':paramY,
    #             'sigma_x':sig_x,'sigma_y':sig_y}
        X = data['X']
        Y = data['Y']
    #     print('empirical var X_t',np.var(X))
    #     print('empirical var Y_t',np.var(Y))
        res = spectral_hsic_test(X, Y, param = np.array([sig_x,sig_y]), nb_eigvals= 5 )
        print('res spectral distrib', res['pval'])
        
    #     # shift_t = RFF_shift_test_roll(X,Y, param = np.array([params['sigma_x'], params['sigma_y']]), approx_min_lag= 'estimated', nb_features=10)
    #     # print('res shift test', shift_t['pval'])
        
    #     # Compute autocorrelation function
    #     combined = X + Y
    #     acf_values = acf(combined, nlags=50, fft=True)
    #     smallest_acf = np.where(acf_values < 0.2)[0] ## first value such that the dependence is under 0.3 (arbitrary)
    #     if len(smallest_acf) == 0:
    #         As = 10
    #     elif smallest_acf[0] > 20:
    #         As = 20
    #     else :
    #         As = np.minimum(10,smallest_acf[0])

        
    #     kernel = 'matern3'
    #     ### Compute Gram matrices of kx and ky from X and Y
    #     Lx = compute_Gram_matrix(params['sigma_x'], X, kernel = kernel)
    #     Ly = compute_Gram_matrix(params['sigma_y'], Y, kernel = kernel)

    #     ## Center Gram matrices
    #     Lx_centered = center_matrix(Lx)
    #     Ly_centered = center_matrix(Ly)
        
    #     ###### COMPARE WITH OTHER METHODS AND PLOT HISTOGRAMS ###########
    #     ## nb_samples: number of samples to estimate the null distribution
    #     nb_samples = 500
    #     ### Compare with Shift-HSIC
        # shift_t = shift_test(X,Y, param = np.array([paramX, paramY]), approx_min_lag= 'median', kernel = kernel, nb_shifts = 500)
        # T_H0 = shift_t['test_stat_H0']
        
        
        
    #     ## asymptotic distribution using true eigenvalues and eigenfunctions in case of iid gaussian variables
    #     # T_asymptotic_rbf = [get_asymptotic_distrib(X, Y,alpha, beta,  params, Sigma = None, r = 5, ts = ts) for _ in range(nb_samples)]
      
        

    #     ## Compare with i.i.d. Gamma distribution 
    #     from scipy.stats import gamma, pearson3

    #     T_gamma = asymptotic_test(X, Y, param = np.array([params['sigma_x'], params['sigma_y']]), kernel = kernel)
    #     large_gamma_param = T_gamma['params']
    #     xx= np.linspace(0,np.max(T_H0)+shift_t['test_statistic'],nb_samples)
                
    #     pdf_large_gamma = gamma.pdf(xx, a = large_gamma_param['shape'], scale = large_gamma_param['scale'])
    #     # r_list = [5]
    #     r_list = [2,3,4,5,8,10,20] ## nb of eigenvalues to consider (for RBF kernels they decrease exponentially so take only first ones)
    #     for r in r_list:
    #         ## Estimate eigenvalues and eigenfunctions
    #         mu_x, V_x = estimate_eigvalues(Lx_centered, r)
    #         mu_y, V_y = estimate_eigvalues(Ly_centered, r)
            
    #         ## compute covariance matrix
    #         if ts :
    #             Sigma = covariance_matrix(V_x, V_y, As)
                
    #         else:
    #             Sigma = np.eye(r**2)
    #             X_t = np.random.uniform(0,0.5, size = (sample_size, ))
    #             Y_t = np.random.normal(size = (sample_size, ))



    #         ## Compute estimated asymptotic distribution of the HSIC
            
    #         spectral_distrib = [1/sample_size * get_asymptotic_sum(mu_x, mu_y, Sigma) for _ in range(nb_samples) ]
            
            
    #         ## PLOT ##
    #         plt.figure(figsize=(12,6))
            
    #         plt.plot(xx, pdf_large_gamma, color = 'red', label = 'pdf of asymptotic approx')
            
    #         plt.hist(T_H0, density = True, alpha=0.6, color='green', label='Shift-HSIC')
    #         # plt.hist(T_asymptotic_rbf, density = True, alpha = 0.4, label = 'spectral distribution')
    #         plt.hist(spectral_distrib, density = True, alpha = 0.6, label = 'estimated spectral distribution' )
            
    #         plt.legend(fontsize = 12)
    #         plt.savefig(save_fig_path + f'/spectral_distrib_n{sample_size}_r{r}_alpha{alpha}_kernel{kernel}.pdf', format = 'pdf')
    #     plt.show()
    # phi = eigenfunction(Lx, v_x, mu_x)
    # plt.figure(figsize=(10,6))

    # for i in range(r):
    #     plt.plot(phi[:,i], label = f'eigenfunction nb {i}')
    # plt.legend()
    # plt.show()

    ## compute covariance matrix
    # Sigma = covariance_matrix(v_x.transpose(),v_y.transpose(), As)
    
    ##check eigenvalues of Sigma
    # eigvals = np.linalg.eigvalsh(Sigma)
    # print("eigenvalues:", eigvals)
    
    
    # asympt = get_asymptotic_sum(data['X'], data['Y'], param = np.array([0.5,0.5]), r = 5)
    # print(asympt)