'''
authors: AA

obj: simulate time series under several dependence scenarios:
1) X_t independent Y_t
2) X_t causes Y_t
3) X_t and Y_t share a common confounder
'''

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform


def scenario_i(l):
    '''
    scenario i samples data from a joint distribution defined 
    the density function f_l(x,y) = 1/(4*pi**2)[1+sin(l*x)sin(l*y)],
    for (x,y) ~ U([-pi,pi]^2)
    Note that as l increases, the dependence becomes more localised
    while marginal densities are uniform on [-pi,pi], \forall l
    '''

    def f_density(l, x, y):
            return (1 /(4 * np.pi**2))*(1+np.sin(l* x)*np.sin(l*y))

    X = np.zeros(1)
    Y = np.zeros(1)
    i = 0
    while i !=1 :
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
    return X,Y


def scenario_ii(l):
    '''
    for l \in \mathbb N, L ~ U({1,...,l})
    \theta ~ U[0,2\pi] and \epsilon_1,\epsilon_2 ~N(0,1)
    returns :
    X = L cos(\theta) + \epsilon_1/4
    Y = L sin(\theta) + \epsilon_2/4

    Note that for large values of l, distribution of (X/l,Y/l)
    approaches a uniform distribution of unit disc
    '''
    L = np.random.choice(np.arange(1, l + 1), 1)
    Theta = np.random.uniform(0, 2 * np.pi, 1)
    epsilon_1 = np.random.normal(0, 1, 1)
    epsilon_2 = np.random.normal(0, 1, 1)

    # Simulate X and Y
    X = L * np.cos(Theta) + epsilon_1 / 4
    Y = L * np.sin(Theta) + epsilon_2 / 4
    return X,Y

def scenario_iii(rho):
    '''
    For independent X and \epsilon, with X ~ U([-1,1])
    and \epsilon ~N(0,1), for \rho \in [0,+\infty)
    return :
    Y = |X|**rho * \epsilon

    This is a global dependence setting between X and Y
    '''
    X = np.random.uniform(low = -1, high = 1, size =1)
    epsilon = np.random.normal(loc = 0, scale = 1, size = 1)
    Y = np.abs(X)**rho * epsilon
    return X,Y


def extinct_gaussian(p,r):
    '''
    \epsilon,\eta following extinct gaussian distribution
    p is the extinction rate (\in [0,1])
    r is the radius of the innovation process 
    see Chwialkowski and Gretton 2014 for more details
    if p goes to 0, the distribution convergers to a Normal(0,1)
    '''
    stop = False
    while stop == False:
        eta = np.random.randn()
        epsilon = np.random.randn()
        d = np.random.uniform()
        if (eta**2 + epsilon**2 > r**2) or (d > p):
            return eta, epsilon


def white_gaussian(sigma):
    eps = np.random.normal(scale = sigma)
    return eps




def update_variables(method,parameters ):
    
    if method == 'scenario_i':
        return scenario_i(**parameters)
    elif method == 'scenario_ii':
        return scenario_ii(**parameters)
    elif method == 'scenario_iii':
        return scenario_iii(**parameters)
    elif method == 'extinct_gaussian':
        return extinct_gaussian(**parameters)
    elif method == 'gaussian':
        return white_gaussian(**parameters)
    else:
        print('!! No valid update method name !!')
        

def generate_random_process(sample_size, alpha, param, noise = 'gaussian'):
    '''
    parameters:
        alpha : float, auto-correlation coefficient
        sample_size : int, number of observations
        param = parameter for the noise
        noise : "", additive noise simulation method
    return:
        X_t = alpha * X_t-1 + eps_t
    
    '''
    X = np.zeros((sample_size, ))
    
    X[0] = np.random.uniform(0, high = 0.5)
    
    for t in range(1,sample_size):
        
        eps = update_variables(noise, parameters = param)
        X[t] = alpha * X[t-1] + eps
    return X


def f_nlinear(rand_int,val):
    '''
    
    Parameters
    ----------
    rand_int : a random integer between 0 and 4
    val : the value of which we apply the chosen function on

    Returns
    -------
    one of the following function chosen : [absolute value, tanh, sin, square]

    '''
    if rand_int <0 or rand_int >4:
        print('rand_int out of range')
        return 0
    if rand_int == 0:
        # distx = cdist(var, var, 'sqeuclidean')
        # var = np.reshape(var,(len(var),1))
        # distx = pdist(var, 'sqeuclidean')
        # return np.exp(-distx)
        return np.exp(-np.sqrt(val**2))
        # return np.abs(val)
    
    if rand_int ==1:
        
        return np.tanh(val)
    
    if rand_int ==2:
        
        return np.sin(val)
    if rand_int==3 :
        
        return val**2
    
    if rand_int == 4:
        return 0.5* val

def f_for_param(rand_int):
    '''
    function only to write in parameters
    Parameters
    ----------
    rand_int : a random integer between 0 and 4

    Returns
    -------
    non linear function selected

    '''
    if rand_int <0 or rand_int >4:
        print('rand_int out of range')
        return 0
    if rand_int == 0:
        # res = 'abs'
        res = 'exp'
    
    if rand_int ==1:
        res = 'tanh'
        
    
    if rand_int ==2:
        res = 'sin'
        
    if rand_int==3 :
        res = 'square'
    

    if rand_int == 4:
        res = 'linear'

    return res

    
if __name__== '__main__':
    param = {'sigma' : 0.5}
    X = generate_random_process(100, 0.5 , param, 'gaussian')
    