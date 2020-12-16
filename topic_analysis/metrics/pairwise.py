import numpy as np
from sklearn.metrics import pairwise_distances


#################################################
# Kernels
#################################################


def cooccurrence_kernel(x, y):
    '''Co-occurrence'''
    return ((x * y).sum() / min(x.sum(), y.sum()))


def npmi(x, y):
    '''Normalized Pointwise Mutual Information score'''
    eps, N = 1, x.shape[0]
    x_and_y = eps + (x * y).sum()
    denominator = eps + x.sum() * y.sum()
    return np.log(N * x_and_y / denominator) / -np.log(x_and_y / N)


def pnpmi(x, y):
    '''Positive Normalized Pointwise Mutual Information score'''
    return max(0, npmi(x, y))


PAIRWISE_KERNEL_FUNCTIONS = {
    'cooccurrence': cooccurrence_kernel,
    'nmpi': npmi,
    'positive_npmi': pnpmi
}


#################################################
# Pairwise distances
#################################################


def cooccurrence_distance(X, discount=False):
    """
    Parameters
    ----------
    X : np.ndarray of shape (n_sample, n_features)

    discount : bool, default=False
        If True, then increase the co-occurrence distance
        between sample A and B proportional to the number
        of samples both A and B independently co-occur with.
    """
    X = 1. - pairwise_distances(X, metric=PAIRWISE_KERNEL_FUNCTIONS['cooccurrence'])
    if discount:
        # TODO: check if this works correctly
#         print(X.min(), X.max(), X.mean(), np.median(X))
        per_phrase_n_cooccurence = (X < 1.).astype(np.int8).sum(0)
        per_phrase_n_cooccurence = np.broadcast_to(
            (X.shape[0] - per_phrase_n_cooccurence) /
            (X.shape[0] - per_phrase_n_cooccurence.min()), X.shape)
        X = np.multiply(X, per_phrase_n_cooccurence)
#         print(X.min(), X.max(), X.mean(), np.median(X))
    return X


PAIRWISE_DISTANCE_FUNCTIONS = {
    'cooccurrence_distance': cooccurrence_distance
}
