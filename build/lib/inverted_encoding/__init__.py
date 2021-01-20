__version__ = '0.0.1'

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

## Main functions
def IEM(trialbyvoxel,features,nfolds=10,stim_max=360,
                      nchannels=9,channel_sd=None,channel_mus=None,
                      is_circular=True,plot_basis_set=False):
    """
    returns predictions, confidences, recons

    This is a longer explanation, which may include math with latex syntax
    :math:`\\alpha`.

    :param trialbyvoxel: the first value
    :param features: the first value
    :param nfolds: the first value
    :type arg1: int, float,...
    :type arg2: int, float,...
    :type arg3: int, float,...
    :returns: arg1/arg2 +arg3

    :Example:

    >>> from inverted_encoding import IEM
    >>> predictions, confidences, recons = IEM(trialbyvoxel,features,stim_max=180,is_circular=True)
    """

    np.random.seed(1) # make procedure reproducible by setting rng
    ntrials = len(features)
    
    # ensure stimulus features are integers less than stim_max
    features = features.astype(int)
    if is_circular:
        features[features==stim_max]=0
    elif np.max(features)==stim_max:
        raise Exception('all features must be in range 0 to stim_max-1')
    if np.any(features>stim_max):
        raise Exception('all features must be less than stim_max')
    
    if type(stim_max)!=int:
        raise Exception('stim_max must be integer')

    # prep basis set, sd = cosine raised to nchannel-1
    if channel_sd==None:
        sd = find_vm_sd(np.cos(np.deg2rad(np.arange(180)))**(nchannels-(nchannels%2)))
        sd = sd * stim_max/360
    else:
        sd = channel_sd
    if channel_mus==None:
        est_mus = np.linspace(0,(stim_max-(stim_max/nchannels)),nchannels).astype(int)
    else:
        est_mus = channel_mus
    
    # optionally show basis set
    if plot_basis_set:
        if is_circular:
            plt.plot(make_gaussian_iter(est_mus,sd*np.ones(stim_max),stim_max))
        else:
            plt.plot(make_noncirc_gaussian_iter(est_mus,sd*np.ones(stim_max),stim_max))
        print("sd of basis channel: {}".format(sd))
        print("mus of basis channels: {}".format(est_mus))
        plt.title("Basis Set")
        plt.xlabel("Stimulus space")
        plt.ylabel("Channel response")
        plt.show()

    # define empty variables
    predictions = np.full(ntrials,np.nan)
    confidences = np.full(ntrials,np.nan)
    recons = np.full((ntrials,stim_max),np.nan)

    kf = KFold(n_splits=nfolds)
    kf.get_n_splits(trialbyvoxel)
    for fold, [train_index, test_index] in (enumerate(kf.split(trialbyvoxel))):
        trnf, tstf = features[train_index], features[test_index]
        trn, tst = trialbyvoxel[train_index,:], trialbyvoxel[test_index,:]

        cr_tsts = np.full((len(tstf),stim_max),np.nan)
        for sh in range(stim_max):
            if ~np.any(np.isnan(cr_tsts)):
                break
            if is_circular:
                basis_set = make_gaussian_iter(est_mus+sh,sd*np.ones(stim_max),stim_max)
            else:
                basis_set = make_noncirc_gaussian_iter(est_mus+sh,sd*np.ones(stim_max),stim_max)
            channelweights_per_vox_iter = np.linalg.lstsq(basis_set[trnf,:], trn, rcond=None)[0]
            cr_tsts[:,est_mus+sh] = np.linalg.lstsq(channelweights_per_vox_iter.T , tst.T, rcond=None)[0].T
        if is_circular:
            est_colors = np.argmax(generate_correlation_map(cr_tsts,
                                        make_gaussian_iter(np.arange(stim_max),
                                                   sd*np.ones(stim_max),stim_max)),axis=1)
            precisions = np.array([sp.stats.linregress(make_gaussian_iter([est_colors[t]],
                                                   sd*np.ones(stim_max),stim_max).flatten(),
                                                   cr_tsts[t,:]).rvalue for t in range(len(tstf))])
        else:
            est_colors = np.argmax(generate_correlation_map(cr_tsts,
                                        make_noncirc_gaussian_iter(np.arange(stim_max),
                                                   sd*np.ones(stim_max),stim_max)),axis=1)
            precisions = np.array([sp.stats.linregress(make_noncirc_gaussian_iter([est_colors[t]],
                                                   sd*np.ones(stim_max),stim_max).flatten(),
                                                   cr_tsts[t,:]).rvalue for t in range(len(tstf))])
        predictions[test_index] = est_colors
        confidences[test_index] = precisions
        recons[test_index,:] = cr_tsts
        
    return predictions, confidences, recons

def permutation(features,stim_max=360,num_perm=5000,is_circular=True):
    np.random.seed(1) # make procedure reproducible by setting rng
    ntrials = len(features)
    
    # ensure stimulus features are integers less than stim_max
    features = features.astype(int)
    if is_circular:
        features[features==stim_max]=0
    elif np.max(features)==stim_max:
        raise Exception('all features must be in range 0 to stim_max-1')
    if np.any(features>stim_max):
        raise Exception('all features must be less than stim_max')
    
    if type(stim_max)!=int:
        raise Exception('stim_max must be integer')

    null_maes=np.full(num_perm,np.nan)
    for perm in range(num_perm):
        feature_shuff = np.random.permutation(features)
        if is_circular:
            null_maes[perm] = np.mean(np.abs(circ_diff(feature_shuff,features,stim_max)))
        else:
            null_maes[perm] = np.mean(np.abs(feature_shuff-features))
        
    return null_maes