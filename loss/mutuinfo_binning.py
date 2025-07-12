"""
Optimized Mutual Information-based Binning Module
-------------------------------------------------

This module is a simplified and optimized adaptation of the code from:
https://github.com/boschresearch/imax-calibration

Purpose:
    The goal is to compute optimal bin boundaries for log-odds values using 
    a mutual information maximization objective. The result is suitable for 
    calibration and interpretability tasks such as ECE or histogram binning.

Highlights:
    - Numerical and computational optimizations are applied.

Author: Wenjian Huang    
"""


import numpy as np
from sklearn.utils.extmath import row_norms, stable_cumsum
from sklearn.metrics.pairwise import euclidean_distances
import scipy.sparse as sp
import random

EPS = np.finfo(float).eps  # used to avoid division by zero

def get_mutualinfo_binning(logits, probs, y, num_bins=15, max_samples=100000):
    """
    Compute bin boundaries for log-odds using mutual information maximization.

    Parameters
    ----------
    logits : np.ndarray
        Raw network logits of shape (N, C) where N is the number of samples and C the number of classes.
    probs : np.ndarray
        Corresponding softmax probabilities of shape (N, C).
    y : np.ndarray
        Ground-truth labels in one-hot encoding of shape (N, C).
    num_bins : int, optional
        Number of bins to produce. Default is 15.
    max_samples : int, optional
        Maximum number of samples to use for optimization. If more, random subsampling is performed.

    Returns
    -------
    bin_boundaries : np.ndarray
        A 1D array of bin boundaries in log-odds space, including -inf and +inf at both ends.
    """

    logodds = quick_logits_to_logodds(logits, probs=probs)
    logodds, y = binary_convertor(logodds, y, cal_setting="sCW", class_idx=None)
    if len(logodds)>max_samples:
        RandSample = random.sample(range(len(logodds)), max_samples)
        bin_boundaries = run_imax(logodds[RandSample], y[RandSample], num_bins, num_steps=200, init_mode='kmeans', bin_repr_during_optim='pred_prob_based', log_every_steps=100)
    else:
        bin_boundaries = run_imax(logodds, y, num_bins, num_steps=200, init_mode='kmeans', bin_repr_during_optim='pred_prob_based', log_every_steps=100)
    bin_boundaries = np.append(-np.inf,bin_boundaries)
    bin_boundaries = np.append(bin_boundaries, np.inf)
    return bin_boundaries



def run_imax(logodds, y, num_bins=15, p_y_pos=None, num_steps=200, init_mode="kmeans", bin_repr_during_optim="pred_prob_based", log_every_steps=10, logfpath=None, skip_slow_evals=True):
    bin_repr_func = bin_representation_function(logodds, y, num_bins, bin_repr_scheme=bin_repr_during_optim)  # get the sample_based or pred_prob_based representations used during training
    bin_boundary_func = bin_boundary_function()

    if init_mode == "kmeans":
        representations, _ = kmeans_pp_init(logodds[..., np.newaxis], num_bins, 755619, mode='jsd')
        representations = np.sort(np.squeeze(representations))
    elif init_mode == "eqmass" or init_mode == "eqsize" or "custom_range" in init_mode:
        boundaries = nolearn_bin_boundaries(num_bins, binning_scheme=init_mode, x=logodds)
        representations = bin_repr_func(boundaries)
    else:
        raise Exception("I-Max init unknown!")

    for i in range(num_steps):
        # Boundary update
        boundaries = bin_boundary_func(representations)
        # Theta - bin repr update
        representations = bin_repr_func(boundaries)

    return boundaries



def bin_representation_function(logodds, labels, num_bins, bin_repr_scheme="sample_based"):
    """
    Get function which returns the sample based bin representations. The function will take in bin boundaries as well as the logodds and labels to return the representations.

    Parameters
    ----------
    logodds: numpy ndarray
        validation logodds
    labels: numpy logodds
        binary labels

    Returns
    -------
    get_bin_reprs: function
        returns a function which takes in bin_boundaries

    """
    def get_bin_reprs(bin_boundaries):
        return bin_representation_calculation(logodds, labels, num_bins, bin_repr_scheme, bin_boundaries=bin_boundaries)
    return get_bin_reprs


def bin_boundary_function():
    def get_bin_boundary(representations):
        return bin_boundary_update_closed_form(representations)
    return get_bin_boundary




# Bin boundary update
def bin_boundary_update_closed_form(representations):
    """
    Closed form update of boundaries. stationary point when log(p(y=1|lambda)) - log(p(y=0|lambda)) = log(log(xxx)/log(xxx)) term. LHS side is logodds/boundaries when p(y|lambda) modelled with sigmoid (e.g. PPB )
    """
    temp_log = 1. + np.exp(-1*np.abs(representations))
    temp_log[temp_log == 0] = EPS
    logphi_a = np.maximum(0., representations) + np.log(temp_log)
    logphi_b = np.maximum(0., -1*representations) + np.log(temp_log)
    assert np.any(np.sign(logphi_a[1:]-logphi_a[:-1])*np.sign(logphi_b[:-1]-logphi_b[1:]) >= 0.)
    temp_log1 = np.abs(logphi_a[1:] - logphi_a[:-1])
    temp_log2 = np.abs(logphi_b[:-1] - logphi_b[1:])
    temp_log1[temp_log1 == 0] = EPS
    temp_log2[temp_log2 == 0] = EPS
    bin_boundaries = np.log(temp_log1) - np.log(temp_log2)
    bin_boundaries = np.sort(bin_boundaries)
    return bin_boundaries



def kmeans_pp_init(X, n_clusters, random_state, n_local_trials=None, mode='jsd'):
    """Init n_clusters seeds according to k-means++

    Parameters
    ----------
    X : array or sparse matrix, shape (n_samples, n_features)
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=np.float64).

    n_clusters : integer
        The number of seeds to choose

    x_squared_norms : array, shape (n_samples,)
        Squared Euclidean norm of each data point.

    random_state : int, RandomState instance
        The generator used to initialize the centers. Use an int to make the
        randomness deterministic.
        See :term:`Glossary <random_state>`.

    n_local_trials : integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007

    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.
    """
    n_samples, n_features = X.shape
    random_state = np.random.RandomState(random_state)
    centers = np.empty((n_clusters, n_features), dtype=X.dtype)
    center_ids = np.empty((n_clusters,), dtype=np.int64)

    #assert x_squared_norms is not None, 'x_squared_norms None in _k_init'
    x_squared_norms = row_norms(X, squared=True)
    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    #test_id   = random_state.randint(n_samples)
    # assert test_id != center_id:
    center_ids[0] = center_id
    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    if mode == 'euclidean':
        closest_dist_sq = euclidean_distances(centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms, squared=True)
    elif mode == 'kl':
        # def KL_div(logits_p, logits_q):
        #    assert logits_p.shape[1] == 1 or logits_q.shape[1] == 1
        #    return (logits_p - logits_q) * (np.tanh(logits_p/2.) * 0.5 + 0.5) + np.maximum(logits_q, 0.) + np.log(1.+np.exp(-abs(logits_q))) + np.maximum(logits_p, 0.) + np.log(1.+np.exp(-abs(logits_p)))
        closest_dist_sq = KL_mtx(X[:, 0], centers[0]).transpose()
    elif mode == 'ce':
        closest_dist_sq = CE_mtx(X[:, 0], centers[0]).transpose()
    elif mode == 'jsd':
        closest_dist_sq = JSD_mtx(X[:, 0], centers[0]).transpose()
    else:
        raise ValueError("Unknown distance in Kmeans++ initialization")

    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rnd_samples = random_state.random_sample(n_local_trials)
        test1 = random_state.random_sample(n_local_trials)
        rand_vals = rnd_samples * current_pot
        assert np.any(abs(test1 - rnd_samples) > 1e-4)

        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        # Compute distances to center candidates
        if mode == 'euclidean':
            distance_to_candidates = euclidean_distances(X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)
        elif mode == 'ce':
            distance_to_candidates = CE_mtx(X[:, 0], X[candidate_ids, 0]).transpose()
        elif mode == 'kl':
            distance_to_candidates = KL_mtx(X[:, 0], X[candidate_ids, 0]).transpose()
        else:
            distance_to_candidates = JSD_mtx(X[:, 0], X[candidate_ids, 0]).transpose()
        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]
        center_ids[c] = best_candidate
        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]

    return centers, center_ids



# Bin representation code
def bin_representation_calculation(x, y, num_bins, bin_repr_scheme="sample_based", bin_boundaries=None, assigned=None, return_probs=False):
    """
    Bin representations: frequency based: num_positive_samples/num_total_samples in each bin.
        or pred_prob based: average of the sigmoid of lambda
    Function gets the bin representation which can be used during the MI maximization.

    Parameters
    ----------
    x: numpy ndarray
        logodds data which needs to be binned using bin_boundaries. Only needed if assigned not given.
    y: numpy ndarray
        Binary label for each sample
    bin_repr_scheme: strig
        scheme to use to determine bin reprs. options: 'sample_based' and 'pred_prob_based'
    bin_boundaries: numpy array
        logodds bin boundaries. Only needed when assigned is not given.
    assigned: int numpy array
        bin id assignments for each sample

    Returns
    -------
    quant_reprs: numpy array
        quantized bin reprs for each sample

    """
    assert (bin_boundaries is None) != (assigned is None), "Cant have or not have both arguments. Need exactly one of them."
    if assigned is None:
        assigned = bin_data(x, bin_boundaries)

    if bin_repr_scheme == "sample_based":
        quant_reprs = bin_repr_unknown_LLR(y, assigned, num_bins, return_probs)  # frequency estimate of correct/incorrect
    elif bin_repr_scheme == "pred_prob_based":
        quant_reprs = bin_repr_unknown_LLR(to_sigmoid(x), assigned, num_bins, return_probs)  # softmax probability for bin reprs
    else:
        raise Exception("bin_repr_scheme=%s is not valid." % (bin_repr_scheme))
    return quant_reprs



def nolearn_bin_boundaries(num_bins, binning_scheme, x=None):
    """
    Get the bin boundaries (in logit space) of the <num_bins> bins. This function returns only the bin boundaries which do not include any type of learning.
    For example: equal mass bins, equal size bins or overlap bins.

    Parameters
    ----------
    num_bins: int
        Number of bins
    binning_scheme: string
        The way the bins should be placed.
            'eqmass': each bin has the same portion of samples assigned to it. Requires that `x is not None`.
            'eqsize': equal spaced bins in `probability` space. Will get equal spaced bins in range [0,1] and then convert to logodds.
            'custom_range[min_lambda,max_lambda]': equal spaced bins in `logit` space given some custom range.
    x: numpy array (1D,)
        array with the 1D data to determine the eqmass bins.

    Returns
    -------
    bins: numpy array (num_bins-1,)
        Returns the bin boundaries. It will return num_bins-1 bin boundaries in logit space. Open ended range on both sides.
    """
    if binning_scheme == "eqmass":
        assert x is not None and len(x.shape) == 1
        bins = np.linspace(1.0/num_bins, 1 - 1.0 / num_bins, num_bins-1)  # num_bins-1 boundaries for open ended sides
        bins = np.percentile(x, bins * 100, interpolation='lower')  # data will ensure its in Logit space
    elif binning_scheme == "eqsize":  # equal spacing in logit space is not the same in prob space because of sigmoid non-linear transformation
        bins = to_logodds(np.linspace(1.0/num_bins, 1 - 1.0 / num_bins, num_bins-1))  # num_bins-1 boundaries for open ended sides
    elif "custom_range" in binning_scheme:  # used for example when you want bins at overlap regions. then custom range should be [ min p(y=1), max p(y=0)  ]. e.g. custom_range[-5,8]
        custom_range = eval(binning_scheme.replace("custom_range", ""))
        assert type(custom_range) == list and (custom_range[0] <= custom_range[1])
        bins = np.linspace(custom_range[0], custom_range[1], num_bins-1)  # num_bins-1 boundaries for open ended sides
    return bins



def safe_log_diff(A, B, log_func=np.log):
    """
    Numerically stable log difference function. Avoids log(0). Will compute log(A/B) safely where the log is determined by the log_func
    """
    if np.isscalar(A):
        if A == 0 and B == 0:
            return log_func(EPS)
        elif A == 0:
            return log_func(EPS) - log_func(B)
        elif B == 0:
            return log_func(A) - log_func(EPS)
        else:
            return log_func(A) - log_func(B)
    else:
        # log(A) - log(B)
        output = np.where(A == 0, log_func(EPS), log_func(A)) - np.where(B == 0, log_func(EPS), log_func(B))
        output[np.logical_or(A == 0, B == 0)] = log_func(EPS)
        assert np.all(np.isfinite(output))
        return output
    


def to_sigmoid(x):
    """
    Stable sigmoid in numpy. Uses tanh for a more stable sigmoid function.

    Parameters
    ----------
    x : numpy ndarray
       Logits of the network as numpy array.

    Returns
    -------
    sigmoid: numpy ndarray
       Sigmoid output
    """
    sigmoid = 0.5 + 0.5 * np.tanh(x/2)
    assert np.all(np.isfinite(sigmoid)) == True, "Sigmoid output contains NaNs. Handle this."
    return sigmoid


def bin_data(x, bins):
    """
    Given bin boundaries quantize the data (x). When ndims(x)>1 it will flatten the data, quantize and then reshape back to orig shape.
    Returns the following quantized values for num_bins=10 and bins = [2.5, 5.0, 7.5, 1.0]\n
    quantize: \n
              (-inf, 2.5) -> 0\n
              [2.5, 5.0) -> 1\n
              [5.0, 7.5) -> 2\n
              [7.5, 1.0) -> 3\n
              [1.0, inf) -> 4\n

    Parameters
    ----------
    x: numpy ndarray
       Network logits as numpy array
    bins: numpy ndarray
        location of the (num_bins-1) bin boundaries

    Returns
    -------
    assigned: int numpy ndarray
        For each sample, this contains the bin id (0-indexed) to which the sample belongs.
    """
    orig_shape = x.shape
    # if not 1D data. so need to reshape data, then quantize, then reshape back
    if len(orig_shape) > 1 or orig_shape[-1] != 1:
        x = x.flatten()
    assigned = np.digitize(x, bins)  # bin each input in data. np.digitize will always return a valid index between 0 and num_bins-1 whenever bins has length (num_bins-1) to cater for the open range on both sides
    if len(orig_shape) > 1 or orig_shape[-1] != 1:
        assigned = np.reshape(assigned, orig_shape)
    return assigned.astype(int)


def CE_mtx(logits_p_in, logits_q_in):
    logits_p = np.reshape(logits_p_in.astype(np.float64), [logits_p_in.shape[0], 1])
    logits_q = np.reshape(logits_q_in.astype(np.float64), [1, logits_q_in.shape[0]])
    CE_mtx = - logits_q * (0.5 + 0.5*np.tanh(logits_p/2.)) + np.maximum(0., logits_q) + np.log(1. + np.exp(-abs(logits_q)))
    return CE_mtx


def KL_mtx(logits_p_in, logits_q_in):
    logits_p = np.reshape(logits_p_in.astype(np.float64), [logits_p_in.shape[0], 1])
    logits_q = np.reshape(logits_q_in.astype(np.float64), [1, logits_q_in.shape[0]])
    KL_mtx = (logits_p - logits_q) * (0.5 + 0.5*np.tanh(logits_p/2.)) + np.maximum(0., logits_q) + np.log(1. + np.exp(-abs(logits_q))) - np.maximum(0., logits_p) - np.log(1. + np.exp(-abs(logits_p)))
    return KL_mtx


def JSD_mtx(logits_p, logits_q):
    logits_p_a = np.reshape(logits_p.astype(np.float64), [logits_p.shape[0], 1])
    logits_q_a = np.reshape(logits_q.astype(np.float64), [1, logits_q.shape[0]])
    logits_q_a = logits_q_a * 0.5 + 0.5 * logits_p_a
    KL_mtx_a = (logits_p_a - logits_q_a) * (0.5 + 0.5*np.tanh(logits_p_a/2.)) + np.maximum(0., logits_q_a) + \
        np.log(1. + np.exp(-abs(logits_q_a))) - np.maximum(0., logits_p_a) - np.log(1. + np.exp(-abs(logits_p_a)))

    logits_p_b = np.reshape(logits_p.astype(np.float64), [1, logits_p.shape[0]])
    logits_q_b = np.reshape(logits_q.astype(np.float64), [logits_q.shape[0], 1])
    logits_p_b = logits_q_b * 0.5 + 0.5 * logits_p_b
    KL_mtx_b = (logits_q_b - logits_p_b) * (0.5 + 0.5*np.tanh(logits_q_b/2.)) + np.maximum(0., logits_p_b) + \
        np.log(1. + np.exp(-abs(logits_p_b))) - np.maximum(0., logits_q_b) - np.log(1. + np.exp(-abs(logits_q_b)))
    return KL_mtx_a * 0.5 + KL_mtx_b.transpose()*0.5



def bin_repr_unknown_LLR(sample_weights, assigned, num_bins, return_probs=False):
    """
    Unknown Bin reprs. Will take the average of the either the pred_probs or the binary labels.
    Determines the bin reprs by taking average of sample weights in each bin.
    For example for sample-based repr: sample_weights should be 0 or 1 indicating correctly classified or not.
    or for pred-probs-based repr: sample_weights should be the softmax output probabilities.
    Handles reshaping if sample_weights or assigned has more than 1 dim.

    Parameters
    ----------
    sample_weights: numpy ndarray
        array with the weight of each sample. These weights are used to calculate the bin representation by taking the averages of samples grouped together.
    assigned: int numpy array
        array with the bin ids of each sample
    return_probs: boolean (default: True)
        All operations take place in logodds space. Setting this to true will ensure that the values returned are in probability space (i.e. it will convert the quantized values from logodds to sigmoid before returning them)

    Returns
    -------
    representations: numpy ndarray
        representations of each sample based on the bin it was assigned to
    """
    orig_shape = sample_weights.shape
    assert np.all(orig_shape == assigned.shape)
    assert sample_weights.max() <= 1.0 and sample_weights.min() >= 0.0, "make sure sample weights are probabilities"
    if len(orig_shape) > 1:
        sample_weights = sample_weights.flatten()
        assigned = assigned.flatten()

    bin_sums_pos = np.bincount(assigned, weights=sample_weights, minlength=num_bins)  # sum up all positive samples
    counts = np.bincount(assigned, minlength=num_bins)  # sum up all samples in bin
    filt = counts > 0
    prob_pos = np.ones(num_bins)*sample_weights.mean()  # NOTE: important change: when no samples at all fall into any bin then default should be the prior
    prob_pos[filt] = bin_sums_pos[filt] / counts[filt]  # get safe prob of pos samples over all samples
    representations = prob_pos
    if return_probs == False:
        representations = to_logodds(representations)  # NOTE: converting to logit domain again
    return representations


def to_logodds(x):
    """

    Convert probabilities to logodds using:

    .. math::
        \\log\\frac{p}{1-p} ~ \\text{where} ~ p \\in [0,1]

    Natural log.

    Parameters
    ----------
    x : numpy ndarray
       Class probabilties as numpy array.

    Returns
    -------
    logodds : numpy ndarray
       Logodds output

    """
    assert x.max() <= 1 and x.min() >= 0
    numerator = x
    denominator = 1-x
    #numerator[numerator==0] = EPS
    # denominator[denominator==0] = EPS # 1-EPS is basically 1 so not stable!
    logodds = safe_log_diff(numerator, denominator, np.log)  # logodds = np.log( numerator/denominator   )
    assert np.all(np.isfinite(logodds)) == True, "Logodds output contains NaNs. Handle this."
    return logodds

def quick_logits_to_logodds(logits, probs=None):
    """
    Using the log-sum-exp trick can be slow to convert from logits to logodds. This function will use the faster prob_to_logodds if n_classes is large.
    """
    n_classes = logits.shape[-1]
    if n_classes <= 100:   # n_classes are reasonable as use this slow way to get marginal
        logodds = logits_to_logodds(logits)
    else:  # imagenet case will always come here!
        if probs is None:
            probs = to_softmax(logits)
        logodds = probs_to_logodds(probs)
    return logodds


def logits_to_logodds(x):
    """
    Convert network logits directly to logodds (without conversion to probabilities and then back to logodds) using:

    .. math::
        \\lambda_k=z_k-\\log\\sum\\nolimits_{k'\\not = k}e^{z_{k'}}

    Parameters
    ----------
    x: numpy ndarray
       Network logits as numpy array

    axis: int
        Dimension with classes

    Returns
    -------
    logodds : numpy ndarray
       Logodds output
    """
    n_classes = x.shape[1]
    all_logodds = []
    for class_id in range(n_classes):
        logodds_c = x[..., class_id][..., np.newaxis] - custom_logsumexp(np.delete(x, class_id, axis=-1), axis=-1)
        all_logodds.append(logodds_c.reshape(-1))
    logodds = np.stack(all_logodds, axis=1)
    assert np.all(np.isfinite(logodds))
    return logodds


def to_softmax(x, axis=-1):
    """
    Stable softmax in numpy. Will be applied across last dimension by default.
    Takes care of numerical instabilities like division by zero or log(0).

    Parameters
    ----------
    x : numpy ndarray
       Logits of the network as numpy array.
    axis: int
       Dimension along which to apply the operation (default: last one)

    Returns
    -------
    softmax: numpy ndarray
       Softmax output
    """
    z = x - np.max(x, axis=axis, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=axis, keepdims=True)
    softmax = numerator / denominator
    assert np.all(np.isfinite(softmax)) == True, "Softmax output contains NaNs. Handle this."
    return softmax


def probs_to_logodds(x):
    """
    Use probabilities to convert to logodds. Faster than logits_to_logodds.
    """
    x = np.clip(x,a_min=0,a_max=1)
    assert x.max() <= 1 and x.min() >= 0
    logodds = to_logodds(x)
    assert np.all(np.isfinite(logodds))
    return logodds


def custom_logsumexp(x, axis=-1):
    """
    Uses the log-sum-exp trick.

    Parameters
    ----------
    x: numpy ndarray
       Network logits as numpy array

    axis: int (default -1)
        axis along which to take the sum

    Returns
    -------
    out: numpy ndarray
        log-sum-exp of x along some axis
    """
    x_max = np.amax(x, axis=axis, keepdims=True)
    x_max[~np.isfinite(x_max)] = 0
    tmp = np.exp(x - x_max)
    s = np.sum(tmp, axis=axis, keepdims=True)
    s[s <= 0] = EPS  # only add epsilon when argument is zero
    out = np.log(s)
    out += x_max
    return out


def binary_convertor(logodds, y, cal_setting, class_idx):
    """
    Function to convert the logodds data (in multi-class setting) to binary setting. The following options are available:
    1) CW - slice out some class: cal_setting="CW" and class_idx is not None (int)
    2) top1 - max class for each sample: get the top1 prediction: cal_setting="top1" and class_idx is None
    3) sCW - merge marginal setting where data is combined: cal_setting="sCW" and class_idx is None
    """

    if cal_setting == "CW":
        assert class_idx is not None, "class_idx needs to be an integer to slice out class needed for CW calibration setting"
        logodds_c = logodds[..., class_idx]
        y_c = y[..., class_idx] if y is not None else None
    elif cal_setting == "top1":
        assert class_idx is None, "class_idx needs to be None - check"
        top1_indices = logodds.argmax(axis=-1)
        logodds_c = logodds[np.arange(top1_indices.shape[0]), top1_indices]
        y_c = y.argmax(axis=-1) == top1_indices if y is not None else None
    elif cal_setting == "sCW":
        assert class_idx is None, "class_idx needs to be None - check"
        logodds_c = np.concatenate(logodds.T)
        y_c = np.concatenate(y.T) if y is not None else None
    else:
        raise Exception("Calibration setting (%s) not recognized!" % (cal_setting))

    return logodds_c, y_c

