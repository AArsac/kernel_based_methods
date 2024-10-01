'''
here we defined functions to compute the shifted test as in Chwialkowski and Gretton 2014,
coupled with some optimization methods as in El Amri, Marrel 2021
'''

import numpy as np

from scipy import stats,optimize
from scipy.spatial.distance import cdist
# from sklearn.gaussian_process.kernels import RBF

from statsmodels.tsa.stattools import acf

import src.tests.utils as utils




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


def generate_shifted_series(data, A, B, k):
    '''
    generate a shifted series
    '''
    # Ensure the input constraints
    n = data.shape[0]
    assert 0 < A <= B <= n, "Ensure 0 < A <= B <= N"

    # Create an array of indices from 0 to length_S_k - 1
    indices = np.arange(n)

    # Calculate the shited indices
    shifted_indices = (A + k + indices) % n

    # Generate the sequence S_k by indexing into Y_t
    S_k = data[shifted_indices]

    return S_k


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


def simple_shift_test(data_x, data_y, param = [], method = 'median', head = 10, tail = 30, approx_min_lag = 'empirical'):
    '''
    Compute a simple shift test of independence as in Chwialkovski & Gretton 2014
    '''
    print('running: non parametric non optimized shift test')
    if len(param)==0:
        sigma_x = utils.set_bandwidth(data_x, method = method)
        sigma_y = utils.set_bandwidth(data_y, method = method)
        param = np.array([sigma_x,sigma_y])
    
    Lx = utils.compute_Gram_matrix(param[0], data_x)
    Ly = utils.compute_Gram_matrix(param[1], data_y)
    T = compute_V_stat(Lx, Ly)

    ## generate shifted processes
    if approx_min_lag == 'empirical':
        A = head
        B = tail
    elif approx_min_lag == 'estimated':
        A,B = estimate_head_and_tail(data_x, data_y)
    print('autocorr A:', A)
    ## estimate null distribution of test statistic
    T_shift = []
    for k in range(A,B):
        Y_shift = generate_shifted_series(data_y, A, B, k)
        Ly_shift = utils.compute_Gram_matrix(param[1],Y_shift)
        T_s = compute_V_stat(Lx, Ly_shift)
        T_shift.append(T_s)

    ## compute p value

    # pval = (T > T_shift).mean()

    pval =( 1 / (len(T_shift) + 1 ) ) * (1 + sum(T < T_shift)) ## this expression to avoid case of pval = 0 which is not realistic
    results = {'pval': pval, 'test_statistic': T}
    return results


def parametric_shift_test(data_x, data_y, param = [], method = 'median', head = 10, tail = 30, approx_min_lag = 'estimated', ):
    '''
    Compute a simple shift test of independence as in Chwialkovski & Gretton 2014
    method: 'median' or 'empirical' --> How to adjust bandwidths
    approx_min_lag = 'estimated' or 'empirical' --> How to define first and last lag in the shifted process
    '''
    print('running: parametric non optimized shift test')
    if len(param)==0:
        sigma_x = utils.set_bandwidth(data_x, method = method)
        sigma_y = utils.set_bandwidth(data_y, method = method)
        param = np.array([sigma_x,sigma_y])
    Lx = utils.compute_Gram_matrix(param[0], data_x)
    Ly = utils.compute_Gram_matrix(param[1], data_y)
    T = compute_V_stat(Lx,Ly)

    ## generate shifted processes
    if approx_min_lag == 'empirical':
        A = head
        B = tail
    elif approx_min_lag == 'estimated':
        A,B = estimate_head_and_tail(data_x, data_y)

    ## estimate null distribution
    T_shift = []
    for k in range(A,B):
        Y_shift = generate_shifted_series(data_y, A, B, k)
        Ly_shift = utils.compute_Gram_matrix(param[1], Y_shift)
        T_shift.append(compute_V_stat(Lx,Ly_shift))
    
    T_shift = np.array(T_shift)

    # Fit a Pearson3 distrib to the shifted test statistics
    shape, loc, scale = stats.pearson3.fit(T_shift)

    # Compute the p-value for the original test statistic T
    pval = 1 - stats.pearson3.cdf(T, shape, loc, scale)
    results = {'pval': pval, 'test_statistic': T}
    return results






if __name__ == '__main__':

    import os
    import pickle
    import tests.UI_perm_test as UI_perm_test

    method = 'scenario_i'
    file_path = os.path.realpath(__file__)
    data_path = os.path.join(os.path.dirname(os.path.dirname(file_path)),f"outputs/simus/ts/{method}/")
    size = 50
    nexp = 100
    # file_path = os.path.join(data_path+f'n{size}_nexp_{nexp}_alpha_0.2_r1.pkl')
    data_file = os.path.join(data_path+f'{method}_nexp100_size150_IndependentTrue.pkl')
    with open(data_file,'rb') as file:
        data = pickle.load(file)

    X = np.array(data['X'].iloc[0])
    Y = np.array( data['Y'].iloc[0])

    pval, val = simple_shift_test(X, Y, approx_min_lag= 'estimated')
    print('simple',pval, val)

    pval, val = parametric_shift_test(X, Y, approx_min_lag= 'estimated')
    print('param', pval, val)


    pval, val = UI_perm_test.permutation_test(X,Y, np.array([0.5,0.5]), method = 'median', nb_permut = 500)
    print('permutation_test :', pval, val)
