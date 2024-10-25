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

def common_confounder(sample_size, alpha, beta, gamma, paramX, paramY, paramC):
    '''
    '''
    ## generate confounding time series
    
    C = f.generate_random_process(sample_size= sample_size, alpha = gamma, param  = {'sigma':paramC}, noise = 'gaussian')
    
    ## initialize X_t and Y_t
    X_t= np.random.uniform(low = 0.1, high = 0.5, size = ((sample_size,)) )
    Y_t= np.random.uniform(low = 0.1, high = 0.5, size = ((sample_size,)) )
    
    
    ## select which non-linear function to apply on C in each series
    rd_x = np.random.randint(0, high  = 4)
    # rd_x = 4
    fx = f.f_for_param(rd_x)
    print('non linear function for C_x :', fx)
    rd_y = np.random.randint(0, high  = 4)
    # rd_y = 4
    fy = f.f_for_param(rd_y)
    print('non linear function for C_y :', fy)
    
    ## simulate the series 
    for j in range(1,sample_size):
        eps_x = paramX * np.random.normal()
        eps_y = paramY * np.random.normal()
        X_t[j] = alpha * X_t[j-1] + f.f_nlinear(rand_int= rd_x, val= C[j] ) + eps_x
        Y_t[j] = beta * Y_t[j-1] + f.f_nlinear(rand_int = rd_y, val =  C[j])  + eps_y
    
    ## save simulation parameters
    parameters = {'alpha' : alpha, 'beta' : beta,
                  'gamma' : gamma,
                  'param_noiseX' : paramX,
                  'param_noiseY': paramY,
                  'param_noiseC': paramC,
                  'fx' : fx,
                  'fy': fy}
    return {'X' : X_t, 'Y': Y_t, 'params': parameters}

def common_confounder_iid(sample_size, alpha, beta, paramC, update_method):
    '''
    Generate time series X,Y such that 
    X = f_x(X_t-1, C, \epsilon_t)
    Y = f_y(Y_t-1, C, \gamma_t)
    with C an additive noise, making X and Y correlated
    '''
    Xt = np.random.uniform(low = 0.01, high = 0.5, size = ((sample_size,)))
    Yt = np.random.uniform(low = 0.01, high = 0.5, size = ((sample_size,)))
    
    for t in range(1,sample_size):
        eps,eta = f.update_variables(update_method, paramC)
        Xt[t] = alpha * Xt[t-1] + eps #+ np.random.normal(0, scale = paramX)
        Yt[t] = beta * Yt[t-1] + eta #+ np.random.normal(0, scale = paramY)
    
    parameters = {'alpha' : alpha, 'beta' : beta,
                #   'param_noiseX' : paramX,
                #   'param_noiseY': paramY,
                  'param_C': paramC,
                  }    
    return {'X':Xt, 'Y': Yt, 'params': parameters}
    
def simulate_iid_data(num_samples=100,param = {}, type=None):
    '''
    type = string: 'i','ii' or 'iii'
    '''
    
    if type == 'i':
        l = param['l']
        def f_density(l, x, y):
            return (1 /(4 * np.pi**2))*(1+np.sin(l* x)*np.sin(l*y))

        X = np.zeros(num_samples)
        Y = np.zeros(num_samples)
        i = 0
        while i !=num_samples :
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

    if type == 'ii':
        l = param['l']
        L = np.random.choice(np.arange(1, l + 1), num_samples)
        Theta = np.random.uniform(0, 2 * np.pi, num_samples)
        epsilon_1 = np.random.normal(0, 1, num_samples)
        epsilon_2 = np.random.normal(0, 1, num_samples)

        # Simulate X and Y
        X = L * np.cos(Theta) + epsilon_1 / 4
        Y = L * np.sin(Theta) + epsilon_2 / 4




    if type == 'iii':
        rho = param['rho']
        ## simulations
        X = np.random.uniform(low = -1, high = 1, size = num_samples)
        epsilon = np.random.normal(loc = 0, scale = 1, size = num_samples)
        Y = np.abs(X)**rho * epsilon
    
    res = {'X' : X, 'Y' : Y, 'params' : param}
    return res

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    
    
    parameters = {'alpha': 0.5, 'beta':0.42, 
                  'params' : {'sigma' : 0.01}
                }
    
    ts_dict = Xt_inde_Yt(sample_size = 200, alpha = parameters['alpha'], 
                     beta = parameters['beta'], 
                     paramX= 0.1, paramY = 0.21)

    df = pd.DataFrame(ts_dict)
