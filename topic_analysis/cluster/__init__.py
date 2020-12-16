import numpy as np
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from topic_analysis.metrics import cooccurrence_distance, PAIRWISE_KERNEL_FUNCTIONS
from topic_analysis.utils import _logging
import warnings


# Global variables
_default_distance_threshold = .3
_default_linkage = 'average'
logger = _logging.get_logger()


def hierarchical_clustering(X, distance_threshold=_default_distance_threshold, linkage=_default_linkage, **kws):
    """
    Hierarchical agglomerative clustering.

    Parameters
    ----------
    X : numpy.ndarray
        Term-document matrix

    distance_threshold : float, default=.5
        Phrases will be clustered using hierarchical clustering
        such that at the end of the process, each cluster represents
        a topic. The number of clusters is determined by the
        `distance_threshold`. Clusters / Topics will stop merging when
        the distance between two topics is higher than the distance
        threshold. Therefore, lower the threshold, the more clusters
        there will be.

        The distance itself is measured as:
        ```
            <=> 1.0 - phrase pair co-occurence confidence
            <=> 1.0 -        # docs with both phrase A and B
                      (-------------------------------------------)
                     min(# docs with phrase A, # docs with phrase B)
        ```

    linkage : str, default='average'
        One of ('average', 'complete', 'single'). 'ward' linkage is not
        possible in this case. See documentation on sklearn.cluster.AgglomerativeClustering
        for more information.

    Returns
    -------
    y_pred : numpy.ndarray
        predicted cluster assignments

    clusterer : sklearn.cluster.AgglomerativeClustering
        clusterer object
    """
    if not (0. <= distance_threshold <= 1.):
        warnings.warn("'distance_threshold={}' must be in range [0., 1.]. "
                      "'distance_threshold' will be converted to {}.".format(
                          distance_threshold, _default_distance_threshold), UserWarning)
        distance_threshold = _default_distance_threshold
    if linkage not in ('average', 'complete', 'single'):
        warnings.warn("'linkage={}' must be in ('average', 'complete', 'single'). "
                      "'linkage' will be set to '{}'.".format(linkage, _default_linkage))
        linkage = _default_linkage

    kws['n_clusters'] = None
    kws['compute_full_tree'] = True
    kws['distance_threshold'] = distance_threshold
    kws['linkage'] = linkage
    if 'affinity' not in kws:
        from functools import partial
        kws['affinity'] = partial(cooccurrence_distance, discount=True)
    clusterer = AgglomerativeClustering(**kws)
    y_pred = clusterer.fit_predict(X)

    return y_pred, clusterer


def GMM_clustering(X, n_components_range, **kws):
    """
    Gaussian mixture model clustering.

    Parameters
    ----------
    X : numpy.ndarray
        Term-document matrix

    n_components_range : Iterable[int]
        Range of n_components parameter values to test
        using BIC (Bayesian information criterion) to
        pick the best GMM model that fits the data.

    kws : dict
        Keywords arguments for sklearn.mixture.GaussianMixture

    Returns
    -------
    y_pred : numpy.ndarray
        predicted cluster assignments

    clusterer : sklearn.mixture.GaussianMixture
        clusterer object
    """
    lowest_bic = np.infty
    bic = []
    kws.pop('n_components', None)
    kws.pop('covariance_type', None)
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = GaussianMixture(n_components=n_components,
                              covariance_type='full', **kws)
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

    bic = np.array(bic)
    logger.info('BIC:', bic)
    y_pred = best_gmm.fit_predict(X)

    return y_pred, best_gmm


def spectral_clustering(X, n_clusters, **kws):
    """
    Spectral clustering with default positive NPMI kernel.
    NPMI = Normalized Pointwise Mutual Information

    Parameters
    ----------
    X : numpy.ndarray
        Term-document matrix

    n_clusters : int
        Number of clusters

    kws : dict
        Keywords arguments for sklearn.cluster.SpectralClustering

    Returns
    -------
    y_pred : numpy.ndarray
        predicted cluster assignments

    clusterer : sklearn.cluster.SpectralClustering
        clusterer object
    """
    if 'affinity' not in kws:
        kws['affinity'] = PAIRWISE_KERNEL_FUNCTIONS['positive_npmi']
    kws.pop('n_clusters', None)
    clusterer = SpectralClustering(n_clusters=n_clusters, **kws)
    y_pred = clusterer.fit_predict(X)

    return y_pred, clusterer


CLUSTER_FUNCTIONS = {
    'hierarchical': hierarchical_clustering,
    'gmm': GMM_clustering,
    'spectral': spectral_clustering}
