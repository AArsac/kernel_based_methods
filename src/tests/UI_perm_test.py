import numpy as np
# from numpy.linalg import eigh, eigvalsh
# from scipy import stats,optimize
# from scipy.spatial.distance import cdist
# from sklearn.gaussian_process.kernels import RBF

import sys 
import os
# If running as the main script (debug mode)
if __name__ == "__main__":
    # Add the current folder (tests) to the system path for local imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    import utils
else:
    # Normal import when used as part of the larger project
    from src.tests import utils

from scipy.stats import gamma, pearson3

# import src.tests.utils as utils





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

    # Ly_permuted_list = generate_permuted_matrices(Ly,nb_permut)
    Y_perm = utils.generate_unique_permutations(data_y,nb_permut)
    T_perm_list = []
    for j in range(nb_permut):
        Ly_perm = utils.compute_Gram_matrix(param[1],Y_perm[j])
        T_perm = compute_V_stat(Lx,Ly_perm)
        T_perm_list.append(T_perm)
        
    #threshold = np.quantile(T_perm_list, 1-alpha)
    #reject = T > threshold

    pval = (T < T_perm_list).mean()
    
    results = {'pval': pval, 'test_statistic': T, 'test_stat_H0': T_perm_list}
    return results

def asymptotic_test(data_x, data_y, param = [], method = 'median'):
    '''
    parameters :
    - data_x: 1-D array (n x 1)
    - data_y: 1-D array (n x 1)
    - param: np.array([\sigma_x, \sigma_y]) bandwidths parameters for kernel x, kernel y
    - method: 'median' or 'empirical'

    returns:
        dict : {'pval' : pval, 'test_statistic': T}
    
    '''
    if len(param)==0:
        sigma_x = utils.set_bandwidth(data_x, method = method)
        sigma_y = utils.set_bandwidth(data_y, method = method)
        param = np.array([sigma_x,sigma_y])
    Lx = utils.compute_Gram_matrix(param[0], data_x)
    Ly = utils.compute_Gram_matrix(param[1], data_y)
    T = compute_V_stat(Lx,Ly)    
    # param_gamma = utils.get_kappa(Lx, Ly)
    param_gamma = utils.get_asymptotic_gamma_params(Lx, Ly)
    
    pval  = 1 - gamma.cdf(T, a = param_gamma['shape'], scale = param_gamma['scale'])
    
    
    # results = {'pval': pval, 'test_statistic': T, 'test_stat_H0': {'x' : xx, 'pdf': pdf_gamma}, 'gamma_params' : param_gamma }
    results = {'pval': pval, 'test_statistic': T, 'gamma_params': param_gamma }
    
    return results
    
def gamma_non_asympt_test(data_x, data_y, param = [], method = 'median'):
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
    if len(param)==0:
        sigma_x = utils.set_bandwidth(data_x, method = method)
        sigma_y = utils.set_bandwidth(data_y, method = method)
        param = np.array([sigma_x,sigma_y])
    Lx = utils.compute_Gram_matrix(param[0], data_x)
    Ly = utils.compute_Gram_matrix(param[1], data_y)
    T = compute_V_stat(Lx,Ly)    
    
    # Parametric estimation of the p-value
    param_gamma = utils.gamma_approx(Lx, Ly)
    
    #compute p-value
    pval = 1 - gamma.cdf(T, a = param_gamma['shape'], scale= param_gamma['scale'] )
    
        
    results = {'pval': pval, 'test_statistic': T, 'gamma_params': param_gamma }
       
    return results

def RFF_perm_test(data_x, data_y, param = [], method = 'empirical', nb_permut = 500, nb_features = 10):
    '''
    Random Fourier Features estimator of the HSIC 
    p value is computed using a permutation scheme
    
    parameters :
    - data_x: 1-D array (n x 1)
    - data_y: 1-D array (n x 1)
    - param: np.array([\sigma_x, \sigma_y]) bandwidths parameters for kernel x, kernel y
    - method: 'median' or 'empirical' to approximate kernel bandwidths
    - nb_permu: integer, number of permutations for the test
    - nb_features : nb of Random Fourier Features to consider (<< n)
    
    returns:
        dict : {'pval' : pval, 'test_statistic': T, 'test_stat_H0' : T_perm} 
    '''
    if len(param)==0:
        sigma_x = utils.set_bandwidth(data_x, method = method)
        sigma_y = utils.set_bandwidth(data_y, method = method)
        param = np.array([sigma_x,sigma_y])
    Zx = utils.generate_RFF(data_x, D = nb_features, sigma = param[0])  # size n x Dx
    Zy = utils.generate_RFF(data_y, D = nb_features, sigma = param[1])  # size n x Dy
    
    T = utils.compute_RFF_hsic(Zx, Zy)
    
    
    Y_perm = utils.generate_unique_permutations(data_y,nb_permut)
    T_perm_list = []
    for j in range(nb_permut):
        Zy_perm = utils.generate_RFF(Y_perm[j], D = nb_features, sigma = param[1])
        T_perm = utils.compute_RFF_hsic(Zx, Zy_perm)
        T_perm_list.append(T_perm)
        
    #threshold = np.quantile(T_perm_list, 1-alpha)
    #reject = T > threshold

    pval = (T < T_perm_list).mean()
    
    results = {'pval': pval, 'test_statistic': T, 'test_stat_H0': T_perm_list}
    return results


def pearson3_approx(data_x, data_y, param = [], method = 'median'):
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
    if len(param)==0:
        sigma_x = utils.set_bandwidth(data_x, method = method)
        sigma_y = utils.set_bandwidth(data_y, method = method)
        param = np.array([sigma_x,sigma_y])
    Lx = utils.compute_Gram_matrix(param[0], data_x)
    Ly = utils.compute_Gram_matrix(param[1], data_y)
    T = compute_V_stat(Lx,Ly)    
    
    # Parametric estimation of the p-value
    param_pe3 = utils.pearson3_moments_cpt(Lx, Ly)
    
    #compute p-value
    # pval = 1 - pearson3.cdf(T,  skew= param_pe3['gamma'], loc = param_pe3['alpha'], scale= param_pe3['beta']  )
    pval = 1 - pearson3.cdf(T,  skew= param_pe3['skewness'], loc = param_pe3['mean'], scale= param_pe3['variance']  )
    
        
    results = {'pval': pval, 'test_statistic': T, 'params': param_pe3 }
       
    return results
    

if __name__ == '__main__':
    import timeit
    import matplotlib.pyplot as plt
    #l = np.random.randint(low = 1, high = 11)  # Choose any value from 1 to 10
    l = 9
    n = 200
    scenario_i = False
    scenario_ii = True
    if scenario_i :
        sc = 'sc1'
        # Generate L, Theta, epsilon_1, epsilon_2
        L = np.random.choice(np.arange(1, l + 1), n)
        Theta = np.random.uniform(0, 2 * np.pi, n)
        epsilon_1 = np.random.normal(0, 1, n)
        epsilon_2 = np.random.normal(0, 1, n)

        # Simulate X and Y
        X = L * np.cos(Theta) + epsilon_1 / 4
        Y = L * np.sin(Theta) + epsilon_2 / 4
        
    if scenario_ii :
        sc = 'sc2'
        def f_density(l, x, y):
            return (1 /(4 * np.pi**2))*(1+np.sin(l* x)*np.sin(l*y))

        X = np.zeros(n)
        Y = np.zeros(n)
        i = 0
        while i !=n :
            # simulate X, Y samples
            Xp = np.random.uniform(-np.pi, np.pi)
            Yp = np.random.uniform(-np.pi, np.pi)

            # Compute joint density values
            Z1 = f_density(l, Xp, Yp)

            #sample from the density
            u = np.random.uniform()
            if u <= Z1:
                X[i] = Xp
                Y[i] = Yp
                i+=1

    
    

    # Add the current folder (tests) to the system path for local imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_fig_path = os.path.dirname(os.path.dirname(current_dir))+'/plot/complexity/'
    
    
    ## TEST

    B = 500 # number of permutations
    # start = timeit.default_timer()
    # res1 = permutation_test(X,Y,method = 'median')
    # stop = timeit.default_timer()
    # print('for permutation test, pval:', res1['pval'],' test stat:', res1['test_statistic'], 'time :', stop - start)
    
    
    # start = timeit.default_timer()
    # res1 = permutation_test(X,Y,method = 'median',nb_permut = B, approx= 'None')
    # stop = timeit.default_timer()
    # print(res1['pval'], res1['test_statistic'], 'time :', stop - start)
    
    # start = timeit.default_timer()
    # res = RFF_perm_test(X, Y, method = 'median')
    # stop = timeit.default_timer()
    # print('for RFF pval:', res['pval'], 'time:', stop - start)

    
    # Example usage to plot the histogram of permutation test + overlay Gamma PDF
    def plot_histogram_with_gamma(data_x, data_y, method='empricial', nb_permut=500, save = False):
        # Run permutation test with both permutation and Gamma approximation
        hsic_data = permutation_test(data_x, data_y, method=method, nb_permut=nb_permut)
        
        # Extract T_perm_list from the permutation test
        T_perm_list = hsic_data['test_stat_H0']  # This will be the list of permuted test statistics
        T_perm = hsic_data['test_statistic']
        print('mean perm :', np.mean(T_perm_list), 'var T perm:', np.var(T_perm_list))
        # print( 'pval for permutation test:', hsic_data['pval'], 'val :', T_perm)
        
        
        #### RFF HISC ########
        RFF_hsic = RFF_perm_test(X, Y, method = method, nb_features= np.int32(0.2*n), nb_permut= nb_permut)
        
        Trff_list = RFF_hsic['test_stat_H0']
        Trff = RFF_hsic['test_statistic']
        # print('pval for RFF perm test', RFF_hsic['pval'], ' val: ',Trff )
        
        
        ##### Pearson 3 approx ####
        pe3_hsic = pearson3_approx(X,Y, method = method)
        # print('pval for pearson 3 approx', pe3_hsic['pval'])
        pe3_params = pe3_hsic['params']
        
        
        # Now run the Gamma approximation to get the Gamma parameters
        hsic_gamma = gamma_non_asympt_test(data_x, data_y, method=method)
        # print('pval for non asymptotic gamma test:', hsic_gamma['pval'], ' val: ', hsic_gamma['test_statistic'])
         
        ## extract non asymptotic gamma parameters 
        gamma_param = hsic_gamma['gamma_params']
               
        ## Run asymptotic test by fitting a gamma distrib
        hsic_asymptotic = asymptotic_test(data_x, data_y, method = method)
        # print( 'pval for asymptotic test:', hsic_asymptotic['pval'],' val: ', hsic_asymptotic['test_statistic'])
        
        large_gamma_param = hsic_asymptotic['gamma_params']
        
        xx= np.linspace(0,np.max(T_perm_list)+T_perm, nb_permut)
        pdf_large_gamma = gamma.pdf(xx, a = large_gamma_param['shape'], scale = large_gamma_param['scale'])

        skew, loc, scale = pearson3.fit(T_perm_list)
        print(skew, loc, scale)
        ### pdf of pearson 3 approx :
        pdf_pe3 = pearson3.pdf(xx, skew= pe3_params['skewness'], loc = pe3_params['mean'], scale= pe3_params['variance'] )#,  pe3_params['gamma'])
        
        # samples_gamma = gamma.rvs(a = gamma_param['shape'], scale = gamma_param['scale'], size = nb_permut)
        pdf_small_gamma = gamma.pdf(xx, a = gamma_param['shape'], scale = gamma_param['scale'])
        
        # Plot histogram of T_perm_list
        plt.hist(T_perm_list, density = True, alpha=0.6, color='skyblue', label='Permutation Distribution')
        
        # Plot histogram of RFF T perm list
        # plt.hist(Trff_list, density = True, alpha=0.6, color='orange', label='RFF Distribution')
        
        ## plot asymptotic gamma function
        
        plt.plot(xx, pdf_large_gamma, color = 'red', label = 'pdf of asymptotic approx')
        
        # plot non asymptotic gamma curve
        # plt.plot(xx, pdf_small_gamma, color='grey', label='Gamma Approximation (Null Distribution)')
        
        # plot pearson 3 pdf
        plt.plot(xx, pdf_pe3, color = 'purple', label = 'pdf of pearson3 approx')
        
        # plt.hist(samples_gamma, bins = 20, density = False, alpha = 0.6, color = 'red', label = 'gamma samples')
        # Add a vertical line for the observed test statistic
        # T = hsic_gamma['test_statistic']
        # plt.axvline(Trff , color='green', linestyle='--', label=f'Observed T_gamma = {Trff:.4f}')
        # plt.axvline(T_perm, color='yellow', linestyle='--', label=f'Observed T_perm = {T_perm:.4f}')
        
        # Add labels and title
        plt.title('Permutation Distribution with Gamma Approximation (HSIC Test)')
        # plt.xlim(left = 0,right = 2*T+3* gamma_param['scale'])
        plt.xlabel('Test Statistic')
        plt.ylabel('Density')
        
        # Add legend
        plt.legend()
        if save:
            plt.savefig(save_fig_path + f'UI_testing_iid_{sc}_l{l}_B{nb_permut}_n{n}.pdf',format = 'pdf')        
        # Show the plot
        plt.show()


    # Example data
    # Assuming data_x and data_y are your input data arrays
    plot_histogram_with_gamma(X, Y, method='empirical', nb_permut=500, save = False)