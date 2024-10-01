import src.simus.basic_func as f
import numpy as np



def Xt_inde_Yt(sample_size, alpha, beta, paramX, paramY):
    '''
    Function to simulate two time series (X_t, Y_t)_t s.t. X_t and Y_t are independent
    
    a series is simulated as X_t = \alpha X_{t-1} + paramX* \epsilon_t
    where \epsilon_t is a N(0,1)
    
    parameters :
    - sample_size : number of observations
    - alpha : auto correlation coeff for X
    - beta : auto correlation coeff for Y
    - paramX : noise parameter for X
    - paramY : noise parameter for Y
    
    return:
        dictionary {X, Y, parameters}
    
    '''
    
    ## simulate X_t and Y_t
    X_t = np.random.uniform(0,0.5, size = (sample_size, ))
    Y_t = np.random.normal(size = (sample_size, ))
    
    for j in range(1,sample_size):
        X_t[j] = alpha * X_t[j-1] + paramX * np.random.normal()
        Y_t[j] = beta * Y_t[j-1] + paramY * np.random.normal()
    ## save simulation parameters
    parameters = {'alpha' : alpha, 'beta': beta,
                  'param_noiseX': paramX,
                  'param_noiseY':paramY
                  }
    return {'X' : X_t, 'Y': Y_t, 'params': parameters}

def Xt_causes_Yt(sample_size, alpha, beta, paramX, paramY, non_linear = True):
    '''
    Function to simulate two time series (X_t, Y_t)_t 
    s.t. X_t causes Y_t
    
    a series is simulated as X_t = \alpha X_{t-1} + paramX * \epsilon_t^x
    where \epsilon_t is a N(0,1)
    the causality is expressed with a simple non-linear function f:
    Y_t = \beta * Y_{t-1} + f(X_t) + paramY * \epsilon_t^y
    parameters :
    - sample_size : number of observations
    - alpha : auto correlation coeff for X
    - beta : auto correlation coeff for Y
    - paramX : noise parameter for X
    - paramY : noise parameter for Y
    
    return:
        dictionary {X, Y, parameters}
    '''    
    ## select which non-linear function to apply on X in Y_t
    if non_linear:
        rd_y = np.random.randint(low =0, high= 4)
    else :
        rd_y = 4
    fy = f.f_for_param(rd_y)
    
    print('selected function :', fy)
    ## initialize X_t, Y_t
    X_t = np.random.uniform(0.01, high = 0.5, size = (sample_size,))
    Y_t = np.zeros((sample_size))
    Y_t[0] = np.random.uniform(low=0.01, high = 0.5)
    
    ## simulate Y_t
    for j in range(1, sample_size):
        X_t[j] = alpha * X_t[j-1] + paramX * np.random.normal()
        Y_t[j] = beta * Y_t[j-1] + f.f_nlinear(rand_int= rd_y, val= X_t[j] ) + paramY * np.random.normal()
    
    ## save simulation parameters
    parameters = {'alpha' : alpha, 'beta' : beta,
                  'param_noiseX' : paramX,
                  'param_noiseY' : paramY,
                  'f' : fy}
    
    return {'X' : X_t, 'Y': Y_t, 'params':parameters}

def common_confounder(sample_size, alpha, beta, params, noise):
    '''
    '''
    ## generate confounding time series
    C = f.generate_random_process(sample_size= sample_size, alpha = 0.2, param  = {'sigma':0.01}, noise = 'gaussian')
    
    ## initialize X_t and Y_t
    X_t= np.random.uniform(low = 0.1, high = 0.5, size = ((sample_size,)) )
    Y_t= np.random.uniform(low = 0.1, high = 0.5, size = ((sample_size,)) )
    
    
    ## select which non-linear function to apply on C in each series
    rd_x = np.random.randint(0, high  = 4)
    fx = f.f_for_param(rd_x)
    print('non linear function for C_x :', fx)
    rd_y = np.random.randint(0, high  = 4)
    fy = f.f_for_param(rd_y)
    print('non linear function for C_y :', fy)
    
    ## simulate the series 
    for j in range(1,sample_size):
        eps_x = f.update_variables(method = noise, parameters= params)
        eps_y = f.update_variables(method = noise, parameters= params)
        X_t[j] = alpha * X_t[j-1] + f.f_nlinear(rand_int= rd_x, val= C[j] ) + eps_x
        Y_t[j] = beta * Y_t[j-1] + f.f_nlinear(rand_int = rd_y, val =  C[j])  + eps_y
    
    ## save simulation parameters
    parameters = {'alpha' : alpha, 'beta' : beta,
                  'param_noise' : params,
                  'noise': noise,
                  'fx' : fx,
                  'fy': fy}
    return {'X' : X_t, 'Y': Y_t, 'params': parameters}


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    
    
    parameters = {'alpha': 0.4, 'beta':0.2, 
                  'params' : {'sigma' : 0.01}
                }
    
    ts_dict = Xt_inde_Yt(5, alpha = parameters['alpha'], 
                     beta = parameters['beta'], 
                     params = parameters['params'],
                     noise = 'gaussian')

    df = pd.DataFrame(ts_dict)
    print(df.head())