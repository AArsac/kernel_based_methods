import numpy as np

from tigramite.independence_tests.cmiknn import CMIknn

from scipy.special import digamma
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors

def CMI_test(X,Y,Z = None, k = 10, shuffle_neighbors = 5, nb_perm = 500):
    '''
    
    '''
    ## preprocess data
    # X = np.reshape(X,(-1, 1))
    # Y = np.reshape(Y,(-1, 1))
    
        
    ## instantiate class
    cd = CMIknn(mask_type=None, significance='shuffle_test', fixed_thres=None, sig_samples= nb_perm,
                 knn=0.2, shuffle_neighbors=5, verbosity=2)
    
    ## preprocess data
    
    if Z is not None:
        # Z = Z.reshape(-1,1)
        data = np.concatenate((X,Y,Z),axis = 1)
        
        xyz = np.array([0]*np.shape(X)[1]+[1]*np.shape(Y)[1]+[2]*np.shape(Z)[1])
    else:
        
        data = np.concatenate((X,Y), axis= 1)
        xyz = np.array([0]*np.shape(X)[1]+[1]*np.shape(Y)[1])
    
    
    print(data.shape)
        
    # print(xyz)
    val = cd.get_dependence_measure(data.T, xyz)
    pval = cd.get_shuffle_significance(data.T, xyz, value = val)
    
    res = {'pval': pval, 'test_statistic': val}
    return res


# def estimate_mi(x_data, y_data, k=5):
#     """
#         KSG Mutual Information Estimator
#         Based on: https://arxiv.org/abs/cond-mat/0305641
#         x_data: data with shape (num_samples, x_dim) or (num_samples,)
#         y_data: data with shape (num_samples, y_dim) or (num_samples,)
#         k: number of nearest neighbors for estimation
#            * k recommended to be on the order of ~ num_samples/10 for independence testing
#     """
#     assert x_data.shape[0] == y_data.shape[0]
#     num_samples = x_data.shape[0]

#     NN = NearestNeighbors(metric='chebyshev')
#     NN.fit(np.concatenate((x_data.reshape(-1, 1) if x_data.ndim == 1 else x_data,
#                                y_data.reshape(-1, 1) if y_data.ndim == 1 else y_data), axis=1))

#     # estimate entropies
#     radius = NN.kneighbors(n_neighbors=k, return_distance=True)[0]

#     radius = np.nextafter(radius[:, -1], 0)

#     NN.fit(x_data.reshape(-1, 1) if x_data.ndim == 1 else x_data)
#     n_x = np.array([i.size for i in NN.radius_neighbors(radius=radius, return_distance=False)])

#     NN.fit(y_data.reshape(-1, 1) if y_data.ndim == 1 else y_data)
#     n_y = np.array([i.size for i in NN.radius_neighbors(radius=radius, return_distance=False)])

#     return NN.digamma(num_samples) + NN.digamma(k) - np.mean(NN.digamma(n_x+1.) + NN.digamma(n_y+1.))

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


def GCE_test(X,Y, tmax = 1, nb_perm = 500, pvalue = True):
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
    if pvalue :
        pval, null = cd.get_shuffle_significance(data.transpose(), xyz, value = val, return_null_dist=True)
 
        res = {'pval': pval, 'test_statistic': val, 'test_stat_H0':null}
    else:
        res = {'pval': val,'test_statistic': val}
    return res




if __name__ == '__main__':
    # n= 100
    # X = np.random.uniform(low = 0, high = 0.5, size = ((n,)))
    # Y = np.random.uniform(low = 0, high = 0.5, size = ((n,)))
    
    # for j in range(1,n):
    #     X[j] = 0.95*X[j-1] + 0.1*np.random.normal()
    #     Y[j] = 0.95*Y[j-1] + 0.23*np.random.normal()
    
    # # X = np.reshape(X, (n,1))
    # # Y = np.reshape(Y, (n,1))
    # res = CMI_test(X,Y, nb_perm = 500)
    # print('pvalue :',res['pval'])
    
    # res_GCE = GCE_test(X,Y,tmax = 1,nb_perm = 500)
    # print('GCE pval:', res_GCE['pval'])
    random_state = np.random.default_rng(seed=42)
    
    n=1000
    Z = np.random.normal(0,1,n)
    Z2 =  np.random.normal(0,1,n)
    Z = Z.reshape((n,1))
    Z2 = Z2.reshape((n,1))
    ZZ = np.concatenate((Z,Z2), axis= 1)
    X = np.zeros((n,1))
    Y = np.zeros((n,1))
    for j in range(n):
        X[j] = np.tanh(Z[j]) + np.random.normal(0,0.7)
        
        Y[j] = np.exp(-Z[j]**2/2) + np.random.normal(0, 0.8)
        
    res = CMI_test(X,Y,Z, nb_perm = 500)
    print('pvalue :',res['pval'],'val', res['test_statistic'])