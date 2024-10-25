import pandas as pd

import os

import numpy as np

from src.tests.shift_hsic import shift_test, random_shift_test, RFF_shift_test
from src.tests.UI_perm_test import permutation_test, asymptotic_test, gamma_non_asympt_test, RFF_perm_test, pearson3_approx_test
from src.tests.MI_tests import GCE_test

from src.simus.scenarios import Xt_causes_Yt, Xt_inde_Yt, common_confounder, simulate_iid_data

# # from tigramite.pcmci import PCMCI
# from tigramite.independence_tests.cmiknn import CMIknn

from scipy.stats import gamma, pearson3

from joblib import Parallel, delayed

import matplotlib.pyplot as plt

import time



### DATA PARAMETERS ###
path_data = 'data/'
data_type = 'ts'  ## 'iid' or 'ts'
dependence_type = 'independence' ### 'independence' or 'direct_dependence' or 'common_confounder'

current_dir = os.path.dirname(os.path.abspath(__file__))
alpha = 0.95 # autocorrelation for X_t
beta =0.3 # autocorrelation for Y_t
non_linear = True ## to simulate linear/non_linear dependence between the time series



### ALGOS PARAMETERS ####

init = np.array([0.5,.5])
esti_param = 'empirical'
B = 500    ### nb of permutations
nb_features = 20    ## nb features for the RFF estimation
# if n < 100:
#     nb_shifts = 100
# elif n >= 100 and n < 500:
#     nb_shifts = n
# elif n > 500:
#     nb_shifts = 500
nb_shifts = 500

## total list : algo_list = ['shift_test','random_shift_test', 'RFF_shift_test','asymptotic',
#                              'permutation_test', 'gamma_non_asymptotic', 'RFF_perm_test','GCE_test']

algo_list = ['permutation_test','asymptotic', 'shift_test']
# algo_list = ['shift_test','random_shift_test', 'RFF_shift_test','asymptotic','gamma_non_asymptotic']
# algo_list = ['GCE_test']


algo_params = {'shift_test':
               {'param':[], 'method':'empirical','head':10,'tail':30, 'approx_min_lag':'estimated'},
               'random_shift_test':
               {'param':[], 'method':'empirical','head':10,'tail':30, 'approx_min_lag':'estimated', 'nb_shifts' : nb_shifts},
               'RFF_shift_test':
               {'param':[], 'method':'empirical','head':10,'tail':30, 'approx_min_lag':'estimated', 'nb_features': nb_features},
               'parametric_shift_test':
               {'param':[], 'method':'empirical','head':10,'tail':30, 'approx_min_lag':'estimated'},
               'opti_shift_test':
               {'param_init':init,'head':10,'tail':30, 'approx_min_lag':'estimated','Grad':True ,'Hessian':True},
               'parametric_opti_shift_test':
               {'param_init':init,'head':10,'tail':30, 'approx_min_lag':'estimated','Grad':True ,'Hessian':True},
               'permutation_test':
               {'param' : [],'method':esti_param,'nb_permut':B},
               'gamma_non_asymptotic':
                {'param' : [], 'method': esti_param},
                'asymptotic':
                {'param': [], 'method': esti_param},
                'RFF_perm_test':
                {'param':[], 'method': esti_param, 'nb_permut': B, 'nb_features':nb_features},
               'opti_permutation_test':
               {'nb_permut':B,'param_init':init},
               'pearson3_test':
                {'param':[], 'method': esti_param },
                'GCE_test':
                {'tmax':1, 'nb_perm':B}
               }




######################
# RUN ALGO functions #
######################

def run_algo(algo, data_x, data_y, params):
    print(f'now running: {algo}')
    # if algo == 'opti_shift_test':
    #     return opti_shift_test(data_x, data_y, **params)
    if algo == 'shift_test':
        return shift_test(data_x, data_y, **params)
    if algo == 'random_shift_test':
        return random_shift_test(data_x, data_y, **params)
    # elif algo == 'parametric_opti_shift_test':
    #     return parametric_opti_shift_test(data_x, data_y, **params)
    # elif algo == 'parametric_shift_test':
    #     return parametric_shift_test(data_x, data_y, **params)
    elif algo == 'pearson3_test':
        return pearson3_approx_test(data_x, data_y, **params)
    elif algo == 'permutation_test':
        return permutation_test(data_x,data_y, **params)
    elif algo == "asymptotic":
        return asymptotic_test(data_x, data_y, **params)
    elif algo == 'RFF_perm_test':
        return RFF_perm_test(data_x, data_y, **params)
    elif algo == 'RFF_shift_test':
        return RFF_shift_test(data_x, data_y, **params)
    
    elif algo == "gamma_non_asymptotic":
        return gamma_non_asympt_test(data_x, data_y, **params)
    # elif algo == 'opti_permutation_test':
    #     return opti_permutation_test(data_x,data_y,**params)
    elif algo == 'GCE_test':
        return GCE_test(data_x, data_y, **params)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")



def get_res(X, Y):
    '''
    run all algorithms in algo_list
    '''
    results = []
    for algo in algo_list:
        start = time.time()
        res = run_algo(algo, X, Y, algo_params[algo])
        end = time.time()
        tmp = end - start
        print('algorithm:', algo,', pval ', res['pval'], ', val ', res['test_statistic'])
        if 'params' in res:
            res_dict = {'algorithm': algo,
                        'pvalue': res['pval'], 'value':res['test_statistic'],
                        'parameters':res['params'], 'time':tmp}    
        else :
            res_dict = {'algorithm': algo,
                        'pvalue': res['pval'], 'value':res['test_statistic'],
                        'Test_H0':res['test_stat_H0'],'time':tmp }
        results.append(res_dict)
    return results

#################
# GENERATE DATA #
#################

# Define the base directory for results
# RESULTS_DIR = f'results/{data_type}/{dependence_type}/{dependence_type}_sample_size_{n}/'

## Ensure the directory exists
# os.makedirs(RESULTS_DIR, exist_ok=True)
# generate = False
# if generate:
#     # Generate or load data based on dependence type
#     if data_type == 'ts' :
#         if dependence_type == 'independence':
#             data = Xt_inde_Yt(n, alpha=alpha, beta=beta, paramX=0.1, paramY=0.21)
#         elif dependence_type == 'direct_dependence':
#             data = Xt_causes_Yt(n, alpha=alpha, beta=beta, paramX=0.1, paramY=0.21, non_linear=non_linear)
#         elif dependence_type == 'common_confounder':
#             data = common_confounder(n, alpha=alpha, beta=beta, params={'sigma': 0.1}, noise='gaussian')
#     else :
#         data = simulate_iid_data(num_samples= n , param = {'l':9}, type = 'ii')

#     X = data['X']
#     Y = data['Y']

########## RUN ON DATA  #############


# get_res(X,Y)



##############################################################
######### RUN FOR DIFFERENT COEFFS ##########
##############################################################


# Define a function that handles one iteration
def run_iteration(alpha, i, data_path, dependence_type, n):
    # Load the data for the given alpha and iteration i
    
    data = np.load(data_path + f'{dependence_type}_size_{n}_alpha{alpha}_{i}.npz')
    X = data['X']
    Y = data['Y']
    
    # Run the test and get results
    res = get_res(X, Y)
    
    # Append 'alpha' and 'iter' to each result
    res_list = []
    for res_dict in res:
        res_dict['alpha'] = alpha
        res_dict['iter'] = i
        res_list.append(res_dict)
    
    return res_list  # Return the results for this iteration


# #### RUN POWER VS AUTO CORRELATION ######
# Parameters
# autocorr = [0.2, 0.6, 0.95]
# # autocorr = [0.95]
# # n = 2000  # sample size

# data_path = path_data + f'/power/{dependence_type}/'
# n_list= [50,100,200,300,500,1000, 2000]
# for n in n_list:
#     path_results = current_dir+f'/results/ts/power/{dependence_type}/size{n}/'
#     # Ensure the directory exists
#     os.makedirs(path_results, exist_ok=True)

#     # Loop over each alpha value
#     for alpha in autocorr:
#         # Run iterations in parallel using joblib.Parallel and delayed
#         res_list = Parallel(n_jobs=-1, prefer = 'processes')(  # Use all available cores; adjust if needed
#             delayed(run_iteration)(alpha, i, data_path, dependence_type, n) for i in range(100)
#         )
        
#         # Flatten the list of lists into a single list
#         res_list_flattened = [item for sublist in res_list for item in sublist]
        
#         # Convert the results to a DataFrame
#         df_results = pd.DataFrame(res_list_flattened)
        
#         # Save the results to a pickle file
#         df_results.to_pickle(path_results + f'GCE_{dependence_type}_size_{n}_alpha{alpha}.pkl')




###### RUN POWER COMMON CONFOUNDER SCENARIO #########


data_path = path_data + f'/power/{dependence_type}/'


def run_iteration_dd(sigmaC, i, data_path, dependence_type, n):
    '''
    run all algos in algo_list for a signel set of data X,Y such
    that X and Y are subject to confounding
    '''
    data = np.load(data_path + f'{dependence_type}_size_{n}_noiseC{sigmaC}_{i}.npz')
    X = data['X']
    Y = data['Y']
    
    # Run the test and get results
    res = get_res(X, Y)
    
    # Append 'alpha' and 'iter' to each result
    res_list = []
    for res_dict in res:
        res_dict['sigmaC'] = sigmaC
        res_dict['iter'] = i
        res_list.append(res_dict)
    
    return res_list  # Return the results for this iteration




# n_list = [50, 100, 150, 200, 250, 300, 400, 500]
# sigmaC = 0.25
# nexp = 100

# path_results = current_dir + f'/results/ts/power/{dependence_type}/'

# os.makedirs(path_results, exist_ok = True)




# for n in n_list:
#     # Run iterations in parallel using joblib.Parallel and delayed
#     res_list = Parallel(n_jobs=-1, prefer = 'processes')(  # Use all available cores; adjust if needed
#         delayed(run_iteration_dd)(sigmaC, i, data_path, dependence_type, n) for i in range(nexp)
#     )
    
#     # Flatten the list of lists into a single list
#     res_list_flattened = [item for sublist in res_list for item in sublist]
    
#     # Convert the results to a DataFrame
#     df_results = pd.DataFrame(res_list_flattened)
    
#     # Save the results to a pickle file
#     df_results.to_pickle(path_results + f'GCE_{dependence_type}_nexp_{nexp}_size_{n}_sigmaC{sigmaC}.pkl')
#     print ('results saved in ', path_results)
    
###################################################    
###### TYPE I ERROR VS AUTO CORRELATION ###########
###################################################    

# autocorr = [0.2,0.6,0.95]
# data_path = path_data + f'/Type_I_error/{dependence_type}/'
# n_list = [50,100,200,300,400,500,700,1000]
# for n in n_list:
#     path_results = current_dir+f'/results/ts/Type_I_error/size{n}/'
#     # Ensure the directory exists
#     os.makedirs(path_results, exist_ok=True)
#     for alpha in autocorr:
#     # Run iterations in parallel using joblib.Parallel and delayed
#         res_list = Parallel(n_jobs=-1, prefer = 'processes')(  # Use all available cores; adjust if needed
#             delayed(run_iteration)(alpha, i, data_path, dependence_type, n) for i in range(100)
#         )
        
#         # Flatten the list of lists into a single list
#         res_list_flattened = [item for sublist in res_list for item in sublist]
        
#         # Convert the results to a DataFrame
#         df_results = pd.DataFrame(res_list_flattened)
#         # res_list = []
#         # for i in range(100):
#         #     data = np.load(data_path+f'{dependence_type}_size_{n}_alpha{alpha}_{i}.npz')
#         #     X = data['X']
#         #     Y = data['Y']
#         #     res = get_res(X,Y)
#         #     for res_dict in res:
#         #         res_dict['alpha'] = alpha
#         #         res_dict['iter'] = i
#                 # res_list.append(res_dict)
#         # df_results = pd.DataFrame(res_list)
#         ## save results
#         df_results.to_pickle(path_results+f'GCE_{dependence_type}_size_{n}_alpha{alpha}.pkl')
#         print('results saved in', path_results)



######### PLOT ###########

def plot_histogram_with_gamma(data_x, data_y, save = False):

    
    ######### NON PARAMETRIC ESTIMATIONS ################
    
    
    if 'permutation_test' in algo_list:
        ####################
        # Permutation test #
        ####################
        algo = 'permutation_test'
        hsic_data = run_algo(algo, data_x, data_y, algo_params[algo])
            
        # Extract T_perm_list from the permutation test
        T_H0 = hsic_data['test_stat_H0']  # This will be the list of permuted test statistics
        T = hsic_data['test_statistic']

        print('mean permutation T', np.mean(T_H0), 'var permutation T', np.var(T_H0))
        
        print( 'pval for permutation test:', hsic_data['pval'])
        algo_base = 'Permutations'
        # Plot histogram of T_perm_list
        plt.hist(T_H0, density = True, alpha=0.6, color='salmon', label='Permutation Distribution')
    
    if 'shift_test' in algo_list :
        ##############
        # Shift test #
        ##############
        algo = 'shift_test'
        shift_hsic = run_algo(algo, data_x, data_y, algo_params[algo])
        T_H0 = shift_hsic['test_stat_H0']
        Tshift = shift_hsic['test_statistic']
        print( 'pval for shifts test:', shift_hsic['pval'], 'val:', Tshift)
        print('nbs shifts for simple shift :',np.shape(T_H0))
        # Plot histogram of T_perm_list
        plt.hist(T_H0, density = True, alpha=0.6, color='skyblue', label='Shifts Distribution')
        algo_base = 'Shifts'
    
    
    if 'random_shift_test' in algo_list :
        #####################
        # Random shift test #
        #####################
        algo = 'random_shift_test'
        rshift_hsic = run_algo(algo, data_x, data_y, algo_params[algo])
        T_H0 = rshift_hsic['test_stat_H0']
        T_rshift = rshift_hsic['test_statistic']
        print( 'pval for random shifts test:', rshift_hsic['pval'], 'val:', T_rshift)
        # print('nbs random shifts:', np.shape(T_H0))
        # Plot histogram of T_perm_list
        plt.hist(T_H0, density = True, alpha=0.6, color='gold', label='Random shifts Distribution')
        algo_base = 'r_shifts'
    
    if 'RFF_perm_test' in algo_list:
        ########################
        # RFF permutation test #
        ########################
        
        algo = 'RFF_perm_test'
        rff_hsic = run_algo(algo, data_x, data_y, algo_params[algo])
        T_H0 = rff_hsic['test_stat_H0']
        T = rff_hsic['test_statistic']
        
        print('pval for ', algo, ': ', rff_hsic['pval'])
        algo_base = 'RFF'
        plt.hist(T_H0, density = True, alpha=0.6, color='orange', label='RFF perm Distribution')
        
    if 'RFF_shift_test' in algo_list :
        ##############
        # Shift test #
        ##############
        algo = 'RFF_shift_test'
        rff_shift = run_algo(algo, data_x, data_y, algo_params[algo])
        T_H0 = rff_shift['test_stat_H0']
        Trff_shift = rff_shift['test_statistic']
        print( 'pval for RFF shift test:', rff_shift['pval'], 'val :', Trff_shift)
        
        # Plot histogram of T_perm_list
        plt.hist(T_H0, density = True, alpha=0.6, color='violet', label='RFF Shift Distribution')
        algo_base = 'RFF Shift'
    
    xx= np.linspace(0, np.max(T_H0), B)
    
    ##### PARAMETRIC ESTIMATIONS ##########
    
    

    if 'gamma_non_asymptotic' in algo_list:
        #############################
        # Non asymptotic gamma test #
        #############################
        algo = 'gamma_non_asymptotic'
        # Now run the Gamma approximation to get the Gamma parameters
        hsic_gamma = run_algo(algo, data_x, data_y, algo_params[algo])
        print('pval for non asymptotic gamma test:', hsic_gamma['pval'])
        
        ## extract non asymptotic gamma parameters 
        gamma_param = hsic_gamma['params']
        # samples_gamma = gamma.rvs(a = gamma_param['shape'], scale = gamma_param['scale'], size = B)
        # plt.hist(samples_gamma, density = True, alpha = 0.7, color = 'grey', label = 'gamma approx hsit')
        pdf_small_gamma = gamma.pdf(xx, a = gamma_param['shape'], scale = gamma_param['scale'])

        #   plot non asymptotic gamma curve
        plt.plot(xx, pdf_small_gamma, color='black', linestyle='dashed', label='Gamma Approximation (perm))')

    if 'asymptotic' in algo_list :
        ###################
        # asymptotic test #
        ###################
        algo = 'asymptotic'
        ## Run asymptotic test by fitting a gamma distrib
        hsic_asymptotic = run_algo(algo, data_x, data_y, algo_params[algo])
        print( 'pval for asymptotic test:', hsic_asymptotic['pval'])
        
        large_gamma_param = hsic_asymptotic['params']
        
        pdf_large_gamma = gamma.pdf(xx, a = large_gamma_param['shape'], scale = large_gamma_param['scale'])
    
    
        ## plot asymptotic gamma function
        
        plt.plot(xx, pdf_large_gamma, color = 'red', label = 'pdf of asymptotic')
    
    if 'pearson3_test' in algo_list:
        #################
        ### PEARSON 3 ###
        #################
        algo = 'pearson3_test'
        ## Run pearson 3 approximation procedure
        pe_hsic = run_algo(algo, data_x, data_y, algo_params[algo])
        print('pval for pearson 3 approximation:', pe_hsic['pval'])
        
        pe_params = pe_hsic['params']
        
        pdf_pe = gamma.pdf(xx - pe_params['gamma'], a = pe_params['alpha'], scale = pe_params['beta'] )
        
        plt.plot(xx, pdf_pe, color ='black', label = 'pdf of pearson3 approximation (perm)')
            
    ############ FIT Person3 OVER shifts ##########
    # p3skew, p3loc, p3scale = pearson3.fit(rshift_hsic['test_stat_H0'])
    # pdf_pe_shift = pearson3.pdf(xx, p3skew, p3loc, p3scale)
    
    ####### FIT Gamma over shifts #########
    # a_g, loc_g, scale_g = gamma.fit(rshift_hsic['test_stat_H0'])
    # pdf_gamma_shift = gamma.pdf(xx, a = a_g, loc = loc_g, scale = scale_g)
    ################
    ### PLOTTING ###
    ################

   # plot Pearson3 pdf from shifts
    
    # plt.plot(xx, pdf_pe_shift, color = 'gold', label = 'pdf of pearson3 based on shifts')
    
    # plot gamma pdf from shifts
    # plt.plot(xx, pdf_gamma_shift, color = 'pink', linestyle = 'dashed', label = 'pdf of gamma based on shifts')
    # Add a vertical line for the observed test statistic

    # plt.axvline(Tshift, color='green', linestyle='--', label=f'Observed T = {Tshift:.6f}')
    # plt.axvline(Trff_shift, color='gold', linestyle='--', label=f'Observed T_perm = {Trff_shift:.6f}')
    
    # Add labels and title
    # plt.title(f'Null distribution for n = {n}, alpha = {alpha}')

    plt.xlabel('Test Statistic', fontsize = 25)
    plt.ylabel('Density', fontsize = 25)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    
    # Add legend
    plt.legend(fontsize = 25, loc = 'best')
    if save:
        plt.savefig(save_fig_path + f'histogram_{data_type}_n{n}_alpha{alpha}.pdf',format = 'pdf')        
    # Show the plot
    plt.show()


# Example data

save_fig_path = current_dir+'/plot/Type_I_error/'
# Assuming data_x and data_y are your input data arrays
n = 500
i=0

data_path = path_data + f'/Type_I_error/{dependence_type}/'
data = np.load(data_path+f'{dependence_type}_size_{n}_alpha{alpha}_{i}.npz')
X = data['X']
Y = data['Y']
plot_histogram_with_gamma(X, Y, save = False)






# Initialize dictionaries to store p-values, test statistics, and computation times
# val_hsic = {'pval':[], 'val':[],'T_null': [] ,'time': []}
# val_gamma = {'pval':[], 'val':[],'T_null': [] , 'time': []}
# val_shift = {'pval':[], 'val':[], 'time': []}
# val_parametric_shift = {'pval':[], 'val':[], 'time': []}
# val_CMI = {'pval':[], 'val':[], 'time': []}

# for k in range(1): 
#     # Generate or load data based on dependence type
#     if data_type == 'ts' :
#         if dependence_type == 'independence':
#             data = Xt_inde_Yt(n, alpha=0.7, beta=0.64, paramX=0.1, paramY=0.21)
#         elif dependence_type == 'direct_dependence':
#             data = Xt_causes_Yt(n, alpha=0.2, beta=0.4, paramX=0.1, paramY=0.21, non_linear=True)
#         elif dependence_type == 'common_confounder':
#             data = common_confounder(n, alpha=0.6, beta=0.7, params={'sigma': 0.1}, noise='gaussian')
#     else :
#         data = simulate_iid_data(num_samples= n , param = {'l':2}, type = 'ii')
        
#     X = data['X']
#     Y = data['Y']

#     # HSIC test with timing
#     start_time = time.time()  # Start timer
#     res_hsic = permutation_test(X, Y, method='median', nb_permut=700)
#     end_time = time.time()  # End timer
#     elapsed_time_hsic = end_time - start_time  # Calculate elapsed time
    
#     print('for HSIC test, pval =', res_hsic['pval'], 'test statistic value =', res_hsic['test_statistic'], 'time =', elapsed_time_hsic)
    
#     val_hsic['pval'].append(res_hsic['pval'])
#     val_hsic['val'].append(res_hsic['test_statistic'])
#     val_hsic['T_null'].append(res_hsic['test_stat_H0'])
#     val_hsic['time'].append(elapsed_time_hsic)
    
#     # Gamma approximation 
#     start_time = time.time()  # Start timer
#     res_gamma = permutation_test(X,Y, approx = 'gamma')
#     end_time = time.time()  # End timer
#     elapsed_time_gamma = end_time - start_time  # Calculate elapsed time
    
#     print('for gamma test, pval =', res_gamma['pval'], 'test statistic value =', res_gamma['test_statistic'], 'time =', elapsed_time_gamma)
    
#     val_gamma['pval'].append(res_gamma['pval'])
#     val_gamma['val'].append(res_gamma['test_statistic'])
#     val_gamma['T_null'].append(res_gamma['test_stat_H0'])
#     val_gamma['time'].append(elapsed_time_gamma)

    # # Shift test 
    # start_time = time.time()
    # res_shift = simple_shift_test(X, Y, method='median', approx_min_lag='estimated')
    # end_time = time.time()
    # elapsed_time_shift = end_time - start_time

    # print('for Shift HSIC test, pval =', res_shift['pval'], 'test statistic value =', res_shift['test_statistic'], 'time =', elapsed_time_shift)
    
    # val_shift['pval'].append(res_shift['pval'])
    # val_shift['val'].append(res_shift['test_statistic'])
    # val_shift['time'].append(elapsed_time_shift)

    # # Parametric shift test 
    # start_time = time.time()
    # res_parametric_shift = parametric_shift_test(X, Y, method='median', approx_min_lag='estimated')
    # end_time = time.time()
    # elapsed_time_parametric_shift = end_time - start_time

    # print('for Parametric Shift test, pval =', res_parametric_shift['pval'], 'test statistic value =', res_parametric_shift['test_statistic'], 'time =', elapsed_time_parametric_shift)

    # val_parametric_shift['pval'].append(res_parametric_shift['pval'])
    # val_parametric_shift['val'].append(res_parametric_shift['test_statistic'])
    # val_parametric_shift['time'].append(elapsed_time_parametric_shift)

    # # CMI test 
    # cd = CMIknn(mask_type=None, significance='shuffle_test', fixed_thres=None, sig_samples=1000,
    #             sig_blocklength=3, knn=int(0.02 * len(X)), shuffle_neighbors=5, confidence='bootstrap', conf_lev=0.9, 
    #             conf_samples=10000, conf_blocklength=1, verbosity=0)

    # start_time = time.time()
    # val = cd.get_dependence_measure(np.array((X, Y)), xyz=np.array([0, 1]))
    # pvalue = cd.get_shuffle_significance(np.array((X, Y)), np.array([0, 1]), val)
    # end_time = time.time()
    # elapsed_time_CMI = end_time - start_time

    # print('for CMI p-value:', pvalue, 'value =', val, 'time =', elapsed_time_CMI)

    # val_CMI['pval'].append(pvalue)
    # val_CMI['val'].append(val)
    # val_CMI['time'].append(elapsed_time_CMI)

# # Save results to compressed .npz files
# np.savez_compressed(os.path.join(RESULTS_DIR, 'hsic_results.npz'), pval=val_hsic['pval'], val=val_hsic['val'], T_null = val_hsic['T_null'] ,time=val_hsic['time'])
# np.savez_compressed(os.path.join(RESULTS_DIR, 'gamma_results.npz'), pval=val_gamma['pval'], val=val_gamma['val'], T_null = val_gamma['T_null'], time=val_gamma['time'])
# np.savez_compressed(os.path.join(RESULTS_DIR, 'shift_results.npz'), pval=val_shift['pval'], val=val_shift['val'], time=val_shift['time'])
# np.savez_compressed(os.path.join(RESULTS_DIR, 'parametric_shift_results.npz'), pval=val_parametric_shift['pval'], val=val_parametric_shift['val'], time=val_parametric_shift['time'])
# np.savez_compressed(os.path.join(RESULTS_DIR, 'cmi_results.npz'), pval=val_CMI['pval'], val=val_CMI['val'], time=val_CMI['time'])

# print(f"Results successfully saved in 'results/{dependence_type}/' directory.")