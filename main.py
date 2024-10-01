import pandas as pd

import os

import numpy as np

from src.tests.shift_hsic import simple_shift_test, parametric_shift_test
from src.tests.UI_perm_test import permutation_test
from src.tests.gammatest import gamma_test
from src.simus.scenarios import Xt_causes_Yt, Xt_inde_Yt, common_confounder

from tigramite.pcmci import PCMCI
from tigramite.independence_tests.cmiknn import CMIknn

import time


path_data = 'data/'

dependence_type = 'independence'

n = 700

# Define the base directory for results
RESULTS_DIR = f'results/{dependence_type}/{dependence_type}_sample_size_{n}/'

# Ensure the directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)




# Initialize dictionaries to store p-values, test statistics, and computation times
val_hsic = {'pval':[], 'val':[], 'time': []}
val_gamma = {'pval':[], 'val':[], 'time': []}
val_shift = {'pval':[], 'val':[], 'time': []}
val_parametric_shift = {'pval':[], 'val':[], 'time': []}
val_CMI = {'pval':[], 'val':[], 'time': []}

for k in range(10): 
    # Generate or load data based on dependence type
    if dependence_type == 'independence':
        data = Xt_inde_Yt(n, alpha=0.7, beta=0.64, paramX=0.1, paramY=0.21)
    elif dependence_type == 'direct_dependence':
        data = Xt_causes_Yt(n, alpha=0.2, beta=0.4, paramX=0.1, paramY=0.21, non_linear=True)
    elif dependence_type == 'common_confounder':
        data = common_confounder(n, alpha=0.6, beta=0.7, params={'sigma': 0.1}, noise='gaussian')
    
    X = data['X']
    Y = data['Y']

    # HSIC test with timing
    start_time = time.time()  # Start timer
    res_hsic = permutation_test(X, Y, method='median', nb_permut=500)
    end_time = time.time()  # End timer
    elapsed_time_hsic = end_time - start_time  # Calculate elapsed time
    
    print('for HSIC test, pval =', res_hsic['pval'], 'test statistic value =', res_hsic['test_statistic'], 'time =', elapsed_time_hsic)
    
    val_hsic['pval'].append(res_hsic['pval'])
    val_hsic['val'].append(res_hsic['test_statistic'])
    val_hsic['time'].append(elapsed_time_hsic)
    
    # Gamma approximation 
    start_time = time.time()  # Start timer
    res_gamma = gamma_test(X, Y)
    end_time = time.time()  # End timer
    elapsed_time_gamma = end_time - start_time  # Calculate elapsed time
    
    print('for gamma test, pval =', res_gamma['pval'], 'test statistic value =', res_gamma['test_statistic'], 'time =', elapsed_time_gamma)
    
    val_gamma['pval'].append(res_gamma['pval'])
    val_gamma['val'].append(res_gamma['test_statistic'])
    val_gamma['time'].append(elapsed_time_gamma)

    # Shift test 
    start_time = time.time()
    res_shift = simple_shift_test(X, Y, method='median', approx_min_lag='estimated')
    end_time = time.time()
    elapsed_time_shift = end_time - start_time

    print('for Shift HSIC test, pval =', res_shift['pval'], 'test statistic value =', res_shift['test_statistic'], 'time =', elapsed_time_shift)
    
    val_shift['pval'].append(res_shift['pval'])
    val_shift['val'].append(res_shift['test_statistic'])
    val_shift['time'].append(elapsed_time_shift)

    # Parametric shift test 
    start_time = time.time()
    res_parametric_shift = parametric_shift_test(X, Y, method='median', approx_min_lag='estimated')
    end_time = time.time()
    elapsed_time_parametric_shift = end_time - start_time

    print('for Parametric Shift test, pval =', res_parametric_shift['pval'], 'test statistic value =', res_parametric_shift['test_statistic'], 'time =', elapsed_time_parametric_shift)

    val_parametric_shift['pval'].append(res_parametric_shift['pval'])
    val_parametric_shift['val'].append(res_parametric_shift['test_statistic'])
    val_parametric_shift['time'].append(elapsed_time_parametric_shift)

    # CMI test 
    cd = CMIknn(mask_type=None, significance='shuffle_test', fixed_thres=None, sig_samples=1000,
                sig_blocklength=3, knn=int(0.02 * len(X)), shuffle_neighbors=5, confidence='bootstrap', conf_lev=0.9, 
                conf_samples=10000, conf_blocklength=1, verbosity=0)

    start_time = time.time()
    val = cd.get_dependence_measure(np.array((X, Y)), xyz=np.array([0, 1]))
    pvalue = cd.get_shuffle_significance(np.array((X, Y)), np.array([0, 1]), val)
    end_time = time.time()
    elapsed_time_CMI = end_time - start_time

    print('for CMI p-value:', pvalue, 'value =', val, 'time =', elapsed_time_CMI)

    val_CMI['pval'].append(pvalue)
    val_CMI['val'].append(val)
    val_CMI['time'].append(elapsed_time_CMI)

# Save results to compressed .npz files
np.savez_compressed(os.path.join(RESULTS_DIR, 'hsic_results.npz'), pval=val_hsic['pval'], val=val_hsic['val'], time=val_hsic['time'])
np.savez_compressed(os.path.join(RESULTS_DIR, 'shift_results.npz'), pval=val_shift['pval'], val=val_shift['val'], time=val_shift['time'])
np.savez_compressed(os.path.join(RESULTS_DIR, 'gamma_results.npz'), pval=val_gamma['pval'], val=val_gamma['val'], time=val_gamma['time'])
np.savez_compressed(os.path.join(RESULTS_DIR, 'parametric_shift_results.npz'), pval=val_parametric_shift['pval'], val=val_parametric_shift['val'], time=val_parametric_shift['time'])
np.savez_compressed(os.path.join(RESULTS_DIR, 'cmi_results.npz'), pval=val_CMI['pval'], val=val_CMI['val'], time=val_CMI['time'])

print(f"Results successfully saved in 'results/{dependence_type}/' directory.")