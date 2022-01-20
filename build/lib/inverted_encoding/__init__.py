__version__ = '0.1.9'

## Import packages
import numpy as np
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
## Convenience functions
def normalize(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))
def make_gaussian_iter(mu,sd,stim_max=180):
    if np.isscalar(mu):
        mu=[mu]
    if np.isscalar(sd):
        sd=[sd]
    return np.array([normalize(np.roll(signal.gaussian(stim_max, std=s),m-stim_max//2)) for m,s in zip(mu,sd)]).T
def make_noncirc_gaussian_iter(mu,sd,stim_max=180):
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
    '''
    Returns circular differences between a and b, where r defines the point in number space where values should be wrapped around (wraps values between -r and r).

    Examples
    ----------
    >>> circ_diff(-10,350,r=360)
    array([0.])
    >>> circ_diff([0,10,180],[20,70,120],r=180)
    array([ 20.,  60., -60.])
    '''
    if np.isscalar(a):
        a = np.array([a])
    if np.isscalar(b):
        b= np.array([b])
    if len(a)!=len(b):
        raise Exception ("a and b must be same length")
    diff = np.full(len(a),np.nan)
    for k in np.arange(len(a)):
        diff[k] = b[k] - a[k]
        if diff[k] < -r//2:
            diff[k] = b[k] - a[k] + r
        elif diff[k] > r//2:
            diff[k] = b[k] - a[k] - r
    return diff
    
def rotate(l,dist):
    # this is identical but faster than np.roll() for larger matrices 
    l=l.tolist()
    for i,d in enumerate(dist):
        ll = l[i]
        for _ in range(d):
            ll.append(ll.pop(0))
    return np.array(l)

## Main functions
def IEM(trialbyvoxel,features,stim_max=180,is_circular=True,nfolds=10,
                      nchannels=9,channel_sd=None,channel_mus=None,
                      plot_basis_set=False,calc_confidence=True):
    """
    Input trial by voxel matrix (or trial by electrode/component/etc.) and the array of presented stimuli and returns trial-by-trial predicted stimuli, confidence values (goodness of fits), and (non-aligned) reconstructions.

    Parameters
    ----------
    trialbyvoxel : matrix (ndarray)
        A matrix of brain activations with shape num_trials by num_voxels (or num_electrodes/num_components/etc.)
    features : int, array
        The stimulus features for every trial, in the same order as the trials specified by trialbyvoxel. All features must be within the range of stimulus space specified by stim_max. All features will be converted to integers if not already.
    stim_max : int, optional
        Specifies the range of stimulus space, such that all features are assumed to be within the range of zero to stim_max-1. E.g., if studying color within a 360° stimulus space, you must change this value to 360 and ensure that all features are integers within 0-359. Defaults to 0-179° stimulus space.
    is_circular: bool, optional
        Specifies whether the stimulus space is circular or not. Defaults to True. E.g., for orientation and color, stimulus space is typically circular (170° and 0° are the same distance away from 175°, assuming stim_max=180); however, for spatial location, you would typically not want a circular stimulus space. 
    nfolds : int, optional
        Number of folds of cross-validation. Defaults to 10.
    nchannels : int, optional
        Number of basis channels in basis set. Note that iterative shifting will be applied such that reconstructions will always be of stim_max length, regardless of number of channels specified. Defaults to 9.
    channel_sd : float, optional
        The standard deviation of each channel of the basis set. Defaults to standard deviation of a cosine raised to nchannels-1. Note that IEM() runs noticeably quicker if channel_sd is specified!
    channel_mus : array_like, optional
        The centers of each channel. Defaults to equally spaced channels. Due to iterative shifting, the centers of each channel are inconsequential if assuming equal spacing. However, if you do not want equally spaced channels (e.g., one channel is always farther apart from the other channels), then provide an array of the centers of each channel (in stimulus space). Note that iterative shifting is applied in all cases.
    plot_basis_set: bool, optional
        Optionally plot the basis set by setting to True. Only plots the first iteration of iterative shifting procedure. Defaults to False.
    calc_confidence: bool, optional
        Whether to return goodness of fits or not. If True, IEM() returns (predictions, confidences, recons, aligned_recons). If False, returns (predictions, recons, aligned_recons). Defaults to True.

    Returns
    -------
    predictions : ndarray
        The predicted stimulus for each trial, calculated by looking for the highest correlation coefficient between each trial reconstruction and each of stim_max basis channels (centered at every integer in stimulus space). The center of the basis channel with the highest correlation is taken as the predicted stimulus (this is the "correlation table" method described in Scotti, Chen, & Golomb (in-prep)).
    confidences : ndarray
        The goodness of fit for each trial. That is, the correlation coefficient (Pearson's R) for each trial when performing the "correlation table" procedure to estimate each trial's predicted stimulus. You may choose to exclude some trials with the lowest confidences to potentially increase statistical power.
    recons : matrix (ndarray)
        Returns trial-by-trial reconstructions as a matrix of shape num_trials by stim_max. Note that these reconstructions are not aligned: the maximum point of a single trial reconstruction is generally associated with that trial's predicted color. You can visualize each trial reconstruction by plotting (e.g., plt.plot(recons[0,:]) for the first trial).
    aligned_recons: matrix (ndarray)
        Returns trial-by-trial reconstructions aligned such that the ground-truth stimuli are zero. That is, if reconstructions are perfect, each trial reconstruction should be centered at zero along the x-axis when plotted, e.g., plt.plot(aligned_recons[trial,:]).

    Example
    ----------
    >>> from inverted_encoding import IEM
    >>> predictions, confidences, recons = IEM(trialbyvoxel,features,stim_max=180,is_circular=True)
    """
    ntrials = len(features)
    
    # ensure stimulus features are integers less than stim_max
    if not (issubclass(type(features[0]), np.integer) or type(features[0])==int):
        raise Exception('All features must be integers within range 0 to stim_max-1. If your stimulus space requires decimal precision, you can multiply your array by some order of magnitude, run this function, and then undo the transformation afterwards.')
    if is_circular:
        features[features==stim_max]=0
    elif np.max(features)==stim_max:
        raise Exception('all features must be in range 0 to stim_max-1')
    if np.any(features>stim_max) or np.any(features<0):
        raise Exception('all features must be in range 0 to stim_max-1')
    
    if type(stim_max)!=int:
        raise Exception('stim_max must be integer')

    if nfolds>ntrials:
        raise Exception('number of folds of cross-validation is above the number of trials!')

    if trialbyvoxel.shape[0]!=ntrials:
        raise Exception('number of trials is not consistent between trialbyvoxel and features!')

    # prep basis set
    if channel_sd==None:
        sd = 42 * stim_max//360 #(sd = approximately cosine raised to nchannel-1)
    else:
        sd = channel_sd
    if np.all(channel_mus==None):
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
    if calc_confidence:
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
            if calc_confidence:
                precisions = np.array([sp.stats.linregress(make_gaussian_iter([est_colors[t]],
                                                   sd*np.ones(stim_max),stim_max).flatten(),
                                                   cr_tsts[t,:]).rvalue for t in range(len(tstf))])
        else:
            est_colors = np.argmax(generate_correlation_map(cr_tsts,
                                        make_noncirc_gaussian_iter(np.arange(stim_max),
                                                   sd*np.ones(stim_max),stim_max)),axis=1)
            if calc_confidence:
                precisions = np.array([sp.stats.linregress(make_noncirc_gaussian_iter([est_colors[t]],
                                                   sd*np.ones(stim_max),stim_max).flatten(),
                                                   cr_tsts[t,:]).rvalue for t in range(len(tstf))])
        predictions[test_index] = est_colors
        if calc_confidence:
            confidences[test_index] = precisions
        recons[test_index,:] = cr_tsts
    # align the reconstructions at zero, such that ideally reconstruction is centered at 0 when plotted
    aligned_recons = rotate(recons,features)
    if calc_confidence:
        return predictions, confidences, recons, aligned_recons
    else:
        return predictions, recons, aligned_recons

def permutation(features,stim_max=180,num_perm=5000,is_circular=True):
    """
    Returns a null distribution of mean absolute error (MAE) values.

    Parameters
    ----------
    features : int, array
        The stimulus features for every trial. All features must be within the range of stimulus space specified by stim_max. All features will be converted to integers if not already.
    stim_max : int, optional
        Specifies the range of stimulus space, such that all features are assumed to be within the range of zero to stim_max-1. E.g., if studying color within a 360° stimulus space, you must change this value to 360 and ensure that all features are integers within 0-359. Defaults to 0-179° stimulus space.
    num_perm : int, optional
        The number of iterations for permutation testing (i.e., the size of the null distribution). Defaults to 5,000.
    is_circular: bool, optional
        Specifies whether the stimulus space is circular or not. Defaults to True. E.g., for orientation and color, stimulus space is typically circular (170° and 0° are the same distance away from 180°, assuming stim_max=180); however, for spatial location, you would typically not want a circular stimulus space. 

    Returns
    -------
    null_mae_distribution : ndarray
        The null distribution of mean absolute error (MAE) values for the given array of stimulus features. Can compare your actual MAE from inverted_encoding.IEM() to this null distribution to assess the statistical significance of your MAE. For example, using a one-tailed test, if over 95% of iterations had worse MAE (i.e., higher values), then it can be concluded that there was stimulus-specific information present in the brain region.

    Example
    ----------
    >>> from inverted_encoding import permutation
    >>> null_mae_distribution = permutation(features,stim_max=180,is_circular=True)
    """
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