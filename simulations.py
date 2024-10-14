import os
import numpy as np
from src.simus.scenarios import Xt_inde_Yt, Xt_causes_Yt, common_confounder
import pandas as pd

# Define base directory for saving data
BASE_DIR = 'data/power/'
# BASE_DIR = f"C:/Users/aa270673/Documents/Code/Code_Gabriel/HSIC/data/"

SCENARIO_DIRS = {
    'independence': os.path.join(BASE_DIR, 'independence'),
    'direct_dependence': os.path.join(BASE_DIR, 'direct_dependence'),
    'non_direct_dependence': os.path.join(BASE_DIR, 'non_direct_dependence')
}

csv = False

# List of sample sizes to simulate
# SAMPLE_SIZES = [50,100, 200, 300, 400, 500, 700, 1000,2000]
SAMPLE_SIZES = [20,50,100]
# Example of noise parameters

NOISE_PARAMX = .1
NOISE_PARAMY = .4

NOISE_PARAMS = {
    'sigma': 0.13
}
NOISE_TYPE = 'gaussian'  # Example noise type

# Define autocorrelation coefficients (alpha, beta)
# alpha = 0.9  # Autocorrelation for X_t
# beta = 0.9   # Autocorrelation for Y_t

non_linear = True

# autocorr = [[0.2,0.22],[0.4,0.37],[0.5,0.53],[0.6,0.59],[0.7,0.72],[0.8,0.75],[0.85,0.87],[0.9,0.88],[0.95,0.93],[0.95,0.95],[0.99,0.99]]
autocorr = [[0.2,0.22],[0.6,0.53],[0.95,0.93]]
# Ensure directories exist
def create_directories():
    """Create directories for storing data if they don't exist."""
    for scenario_dir in SCENARIO_DIRS.values():
        os.makedirs(scenario_dir, exist_ok=True)

def save_simulation_data(scenario, sample_size, data,i):
    """Helper function to save simulated data and parameters."""
    folder = SCENARIO_DIRS[scenario]
    alpha = data['params']['alpha']
    filename = f'{scenario}_size_{sample_size}_alpha{alpha}_{i}.csv'
    filepath = os.path.join(folder, filename)
    
    # The dict `data` contains 'X', 'Y', and 'params'
    
    if csv :
        df = pd.DataFrame(columns = ['X','Y','params'])
        df['X'] = data['X']
        df['Y'] = data['Y']
        df['params'] = data['params']
        df.to_csv(filepath)
    else:
        np.savez_compressed(filepath, X=data['X'], Y=data['Y'], params=data['params'])
    print(f"Data saved at: {filepath}")

def simulate_data():
    """Simulate and save data for all scenarios and sample sizes.
    """
    
    # Create directories if they don't exist
    create_directories()
    
    # Iterate over each sample size
    for sample_size in SAMPLE_SIZES:
        # Scenario 1: Independent X_t and Y_t
        for alpha,beta in autocorr :
            for i in range(1):
                data_independent = Xt_inde_Yt(sample_size, alpha, beta, NOISE_PARAMX, NOISE_PARAMY)
                save_simulation_data('independence', sample_size, data_independent,i)

        # Scenario 2: X_t causes Y_t
                # data_direct_dependence = Xt_causes_Yt(sample_size, alpha, beta, NOISE_PARAMX, NOISE_PARAMY, non_linear= non_linear)
                # save_simulation_data('direct_dependence', sample_size, data_direct_dependence,i)

        # Scenario 3: Common confounder
        # data_confounder = common_confounder(sample_size, alpha, beta, NOISE_PARAMS, NOISE_TYPE)
        # save_simulation_data('non_direct_dependence', sample_size, data_confounder)



if __name__ == "__main__":
    simulate_data()
