## Import packages
import numpy as np
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
## Convenience functions
k2sd = lambda k : np.sqrt( -2 * np.log( sp.special.iv(1,k) / sp.special.iv(0,k) ) )
def sd2k(S):
    try:
        len(S)
    except:
        S = np.array([S])
    R = np.exp(-S**2/2)
    K = 1/(R**3 - 4 * R**2 + 3 * R)
    if np.any(R < 0.85):
        K[R < 0.85] = -0.4 + 1.39 * R[R < 0.85] + 0.43/(1 - R[R < 0.85])
    if np.any(R < 0.53):
        K[R < 0.53] = 2 * R[R < 0.53] + R[R < 0.53]**3 + (5 * R[R < 0.53]**5)/6
    if len(S) == 1:
        K = float(K)
    return K
def normalize(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))
def make_gaussian_iter(mu,sd,stim_max=360):
    if np.isscalar(mu):
        mu=[mu]
    if np.isscalar(sd):
        sd=[sd]
    return np.array([normalize(np.roll(signal.gaussian(stim_max, std=s),m-stim_max//2)) for m,s in zip(mu,sd)]).T
def make_noncirc_gaussian_iter(mu,sd,stim_max=360):
    if np.isscalar(mu):
        mu=[mu]
    if np.isscalar(sd):
        sd=[sd]
    return np.array([normalize(sp.stats.norm.pdf(np.arange(stim_max), m, s))
                         for m,s in zip(mu,sd)]).T
def generate_correlation_map(x, y):
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x, y.T) - n * np.dot(mu_x[:, np.newaxis], mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])
def circ_diff(a,b,r=360):
    if np.isscalar(a):
        a = np.array([a])
    if np.isscalar(b):
        b= np.array([b])
    diff = np.full(len(a),np.nan)
    for k in np.arange(len(a)):
        diff[k] = b[k] - a[k]
        if diff[k] < -r//2:
            diff[k] = b[k] - a[k] + r
        elif diff[k] > r//2:
            diff[k] = b[k] - a[k] - r
    return diff
def find_vm_sd(channel_responses,plotting=False):
    # fits to von Mises distribution
    xReal = np.arange(360)
    
    f = sp.interpolate.interp1d(np.arange(len(channel_responses)),channel_responses)
    yReal = f(np.arange(0,len(channel_responses)-1,(len(channel_responses)-1)/360))
    yReal = np.roll(yReal,180-np.argmax(yReal))
    
    losses = [np.abs(np.sum(yReal - make_gaussian_iter(180,sd))) for sd in np.arange(10,180)]

    if plotting:
        plt.plot(yReal)
        plt.plot(make_gaussian_iter(180,np.arange(10,180)[np.argmin(losses)]))
        
    return np.arange(10,180)[np.argmin(losses)]