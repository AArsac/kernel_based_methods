import numpy as np

from tigramite.independence_tests.cmiknn import CMIknn

def CMI_test(X,Y, k = 10, shuffle_neighbors = 5, nb_perm = 500):
    '''
    
    '''
    ## preprocess data
    X = np.reshape(X,(len(X), 1))
    Y = np.reshape(Y,(len(Y), 1))
    
    ## instantiate class
    cd = CMIknn(mask_type=None, significance='shuffle_test', fixed_thres=None, sig_samples= nb_perm,
                 knn=0.2, shuffle_neighbors=5, verbosity=2)
    
    ## preprocess data
    data = np.concatenate((X,Y), axis= 1)
    print(data.shape)
    xyz = np.array([0]*np.shape(X)[1]+[1]*np.shape(Y)[1])
    print(xyz)
    val = cd.get_dependence_measure(data.transpose(), xyz)
    pval = cd.get_shuffle_significance(data.transpose(), xyz, value = val)
    
    res = {'pval': pval, 'test_statistic': val}
    return res


def build_lagged_matrix(data_matrix, max_lag):
    """
    Build the lagged matrix of all the variables as follows :
        Mat (N x K) --> all_lagM (N x (K*Lmax)) 
    [X_0,X_1,...,X_K]-->[X_t^0, X_{t-1}^0,..., X_{t-Lmax}^0, X_t^1,...,X_{t-Lmax}^K]
    """
    N, K = data_matrix.shape
    lagged_matrix = np.zeros((N, K * max_lag))  ##lagged matrix of all variables

    ind_lagged_matrix = np.zeros((K, 2)).astype(int)     # start and end of columns of each variable in lagged matrix
    cpt = 0

    ##  lag starts at 0 !!
    for var in range(K):
        ind_lagged_matrix[var, 0] = cpt
        ind_lagged_matrix[var, 1] = cpt + max_lag

        lagged_matrix[:, ind_lagged_matrix[var, 0]] = data_matrix[:, var]   ## we fill the columns whose lag =0

        for lag in range(1, max_lag):   #for lag = 1...Lmax-1
            lagged_matrix[lag:, ind_lagged_matrix[var, 0] + lag] = data_matrix[:-lag, var]
        cpt += max_lag

    lagged_matrix = lagged_matrix[max_lag - 1:-1, :]    ## to start and end at the same time
    return lagged_matrix, ind_lagged_matrix


def GCE_test(X,Y, tmax = 1, nb_perm = 500):
    '''
    Greedy Causation Entropy, taken from C.K. Assaad
    https://proceedings.mlr.press/v180/assaad22a
    found also on github.com/ckassaad/PCGCE
    '''
    
    ## instantiate class
    cd = CMIknn(mask_type=None, significance='shuffle_test', fixed_thres=None, sig_samples= nb_perm,
                 knn=0.1, shuffle_neighbors=5, verbosity=2)
    
    ## preprocess data
    X = np.reshape(X,(len(X), 1))
    Y = np.reshape(Y,(len(Y), 1))
    
    ## Take lagged components of the input series to take into account autocorrelation
    X_lag, ind = build_lagged_matrix(X, max_lag = tmax)

    ## Reshape Y to match X_lag length
    Y = Y[tmax-1:-1]
    data = np.concatenate((X_lag,Y), axis= 1)
    
    xyz = np.array([0]*np.shape(X_lag)[1]+[1]*np.shape(Y)[1])

    val = cd.get_dependence_measure(data.transpose(), xyz)
    pval, null = cd.get_shuffle_significance(data.transpose(), xyz, value = val, return_null_dist=True)
    
    res = {'pval': pval, 'test_statistic': val, 'test_stat_H0':null}
    return res

if __name__ == '__main__':
    n= 100
    X = np.random.uniform(low = 0, high = 0.5, size = ((n,)))
    Y = np.random.uniform(low = 0, high = 0.5, size = ((n,)))
    
    for j in range(1,n):
        X[j] = 0.95*X[j-1] + 0.1*np.random.normal()
        Y[j] = 0.95*Y[j-1] + 0.23*np.random.normal()
    
    # X = np.reshape(X, (n,1))
    # Y = np.reshape(Y, (n,1))
    res = CMI_test(X,Y, nb_perm = 500)
    print('pvalue :',res['pval'])
    
    res_GCE = GCE_test(X,Y,tmax = 1,nb_perm = 500)
    print('GCE pval:', res_GCE['pval'])