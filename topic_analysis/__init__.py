"""
Various classes and functions for topic analysis on text data.

References
----------
* https://www.researchgate.net/publication/322823957_Trending_Topics_Detection_of_Indonesian_Tweets_Using_BN-grams_and_Doc-p
"""

from .about import __version__
import numpy as np
import pandas as pd
import spacy
from topic_analysis.cluster import CLUSTER_FUNCTIONS
from topic_analysis.feature_extraction import (
    get_ranked_phrases,
    _default_n_jobs,
    _default_batch_size
)
from topic_analysis.utils import _logging
import warnings


# Global variables
SUPPORTED_LANGUAGES = {'nl': 'dutch', 'en': 'english'}
_default_spacy_lang = 'nl_core_news_sm'
_default_clustering_method = 'hierarchical'
logger = _logging.get_logger()


class TopicsAnalyzer:
    """
    Extract topics from input text data.

    Attributes
    ----------
    clusterer_ : Union[sklearn.base.ClusterMixin,
        sklearn.base.DensityMixin]
        Clusterer object used to cluster the phrases.

    topics_ : pandas.DataFrame
        List of topics found sorted in descending maximum
        BNgram score per topic.

    n_topics_ : Union[int, None]
        Number of topics found if `fit_predict` has been
        called else None.

    References
    ----------
    * Mining newsworthy topics from social media.
      Martin, Carlos and Corney, David and Goker, Ayse (2015).
      Advances in social media analysis (Springer), pp 21-43
    """

    def __init__(
            self, nlp, *, include_verb_phrases=True, minlen=1, maxlen=8,
            n_jobs=_default_n_jobs, batch_size=_default_batch_size,
            stop_phrases=[], vectorizer='bngram', vectorizer_kws=dict(),
            clustering_method=_default_clustering_method,
            clustering_kws=dict(), ranking='size', remove_substrings=True):
        """
        Parameters
        ----------
        nlp : spacy.language.Language
            Spacy language model

        include_verb_phrases : bool, default=False
            Indicator to include verb phrases also.

        minlen : int, default=1
            Minimum length of extracted multi-word phrases.
            Used for tokenizing the text.

        maxlen : int, default=8
            Maximum length of extracted multi-word phrases.
            Used for tokenizing the text.

        n_jobs : int, default=-1
            Number of processes to get noun phrases in parallel
            from documents.
                * -1: Use one process per available CPU cores
                * >0: Use `n_jobs` processes

        batch_size : int, default=1000
            Batch size for tokenizing, tagging and extracting
            noun phrases. Use smaller batch sizes on large
            number of large texts and vice-versa.

        stop_phrases : List[str], default=[]
            List of phrases to remove.

        vectorizer : str, default='bngram'
            One of ('bngram', 'tfidf').

        vectorizer_kws : dict, default={}
            Keyword arguments for BNgrams vectorizer.

        clustering_method : string, default='hierarchical'
            One of ('hierarchical', 'gmm', 'spectral').
            * hierarchical: Hierarchical Agglomerative clustering
            * gmm: Gaussian Mixture modeling
            * spectral: Spectral clustering

        clustering_kws : dict, default={}
            Keyword arguments for clustering method.

        ranking : str, default='size'
            Method of ranking topics. One of ('size', 'bngram').
            'size' ranks topics by number of documents and
            'bngram' ranks topics by the maximum bngram score
            of phrases within the topics.

        remove_substrings : bool, default=True
            Phrase extraction may generate a lot of overlapping
            and subtring phrases. If True, then we remove those
            substring phrases from each topic in the final result.
        """
        # assign variables
        self.nlp = nlp
        self.include_verb_phrases = include_verb_phrases
        self.minlen = minlen
        self.maxlen = maxlen
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.stop_phrases = stop_phrases
        self.vectorizer = vectorizer
        self.vectorizer_kws = vectorizer_kws
        self.clustering_method = clustering_method
        self.clustering_kws = clustering_kws
        self.remove_substrings = remove_substrings
        self.ranking = ranking

    def fit_predict(self, raw_documents, timestamps=None, y=None):
        """
        Parameters
        ----------
        raw_documents : Iterable[str]
            An iterable which yields either str objects.

        timestamps : Iterable[datetime], optional
            Timestamp of the documents. An iterable which
            yields datetime objects. Required when `vectorizer='bngram'`.

        y : Iterable
            Not used, only there for sklearn compatibility.

        Returns
        -------
        doc_topic_series : pd.Series[Tuple[int]]
            Document-topic mapping
        """
        self._check_params()

        # extract phrases and vectorize with BNgrams
        X, vectorizer = get_ranked_phrases(
            self.nlp, raw_documents, timestamps,
            include_verb_phrases=self.include_verb_phrases,
            minlen=self.minlen, maxlen=self.maxlen,
            n_jobs=self.n_jobs, batch_size=self.batch_size,
            stop_phrases=self.stop_phrases, vectorizer=self.vectorizer,
            aggfunc=None, **self.vectorizer_kws)
        phrases = np.array(vectorizer.get_feature_names())
        logger.info(f'Extracted {len(phrases)} phrases')
        del raw_documents, timestamps, vectorizer

        # topic clustering
        logger.info(f'Clustering phrases into topics with {self.clustering_method} method')
        doc_term_matrix = (X > 0).astype(np.int8)
        y_pred, clusterer = CLUSTER_FUNCTIONS[self.clustering_method](
            X.T.toarray(), **self.clustering_kws)

        # topic ranking
        if self.ranking == 'size':
            logger.info('Ranking topics based on topic size')
            topic_scores = pd.DataFrame(doc_term_matrix.T.toarray())\
                .groupby(y_pred)\
                .agg(lambda x: np.bitwise_or.reduce(x.values))\
                .sum(axis=1)
        elif self.ranking == 'bngram':
            logger.info('Ranking topics based on max BNgram score per topic')
            topic_scores = pd.DataFrame(min_max_axis(X.tocsc(), axis=0, ignore_nan=True)[1])\
                .groupby(y_pred)\
                .agg(max)[0]
        else:
            raise ValueError(f"`ranking={self.ranking}` must be one of ('size', 'bngram')")
        topics = pd.DataFrame(phrases)\
            .groupby(y_pred)\
            .apply(lambda _: _[0].tolist())\
            .reset_index()\
            .assign(score=topic_scores)\
            .drop('index', axis=1)\
            .rename(columns={0: 'phrases'})\
            .sort_values('score', ascending=False)
        logger.info(f'Found {topics.shape[0]} topics')

        # assign topic id to each document
        logger.info('Mapping documents to topic IDs')
        doc_topic_series = pd.Series([set() for _ in range(doc_term_matrix.shape[0])])
        doc_topic_matrix = doc_term_matrix.multiply((y_pred + 1)[np.newaxis, :]).tocoo()
        for i, j, v in zip(
            doc_topic_matrix.row, doc_topic_matrix.col, doc_topic_matrix.data):
            doc_topic_series[i].add(v - 1)
        doc_topic_series = doc_topic_series.apply(tuple)

        # if any document is assigned no topics
        no_topics = doc_topic_series.apply(len) == 0
        if no_topics.sum() > 0:
            no_topics[no_topics].index.values
            logger.warning(f"Documents {no_topics[no_topics].index.values.tolist()} are "
                           "not assigned any topics.")

        # remove substrings from topics
        if self.remove_substrings:
            def remove_substrings(l):
                l = l[:]
                l.sort(key=len)
                idxs = [i for i, s1 in enumerate(l[:]) if any(s1 in s2 for s2 in l[i+1:])]
                idxs.sort(reverse=True)
                for i in idxs:
                    l.pop(i)
                return l
            topics['phrases'] = topics['phrases'].apply(remove_substrings)
        

        # set attributes
        self.clusterer_, self.topics_ = clusterer, topics

        logger.info('Done')
        return doc_topic_series

    @property
    def n_topics_(self):
        return self.topics_.shape[0]\
        if hasattr(self, 'topics_') else None

    def _validate_nlp(self):
        msg = None
        if not isinstance(self.nlp, spacy.language.Language):
            msg = f"spacy 'nlp <{self.nlp}>' module not initialized correctly."
        elif self.nlp.lang not in SUPPORTED_LANGUAGES:
            msg = f"spacy 'nlp <{self.nlp}>' language not supported."
        if msg:
            msg += f" Will load the spacy {_default_spacy_lang} language module."
            warnings.warn(msg, UserWarning)
            self.nlp = None
            nlp = spacy.load(_default_spacy_lang, disable=['parser', 'ner'])
            self.nlp = nlp

    def _validate_n_jobs(self):
        if not (isinstance(self.n_jobs, int) and
                (self.n_jobs == -1 or self.n_jobs > 0)):
            warnings.warn("'n_jobs' must be -1 or a positive integer. 'n_jobs' will "
                          f"be converted to {_default_n_jobs}", UserWarning)
            self.n_jobs = _default_n_jobs

    def _validate_batch_size(self):
        if not (isinstance(self.batch_size, int) and self.batch_size > 0):
            warnings.warn("'batch_size' must be a positive integer. 'batch_size' "
                          f"will be converted to {_default_batch_size}", UserWarning)
            self.batch_size = _default_batch_size

    def _validate_vectorizer_kws(self):
        if not isinstance(self.vectorizer_kws, dict):
            warnings.warn("'vectorizer_kws' must be a dict. "
                          "'vectorizer_kws' will be set to dict().", UserWarning)
            self.vectorizer_kws = dict()
        if 'norm' in self.vectorizer_kws:
            warnings.warn("'vectorizer_kws' should not contain key 'norm'. "
                          "It will be removed.", UserWarning)
            self.vectorizer_kws.pop('norm', None)
        if 'analyzer' in self.vectorizer_kws:
            warnings.warn("'vectorizer_kws' should not contain key 'analyzer'. "
                          "It will be removed.", UserWarning)
            self.vectorizer_kws.pop('analyzer', None)

    def _validate_clustering_method(self):
        if self.clustering_method not in CLUSTER_FUNCTIONS:
            warnings.warn("Only {} clustering methods supported. 'clustering_method' "
                          "will be set to '{}'.".format(
                              list(CLUSTER_FUNCTIONS.keys()),
                              _default_clustering_method), UserWarning)
            self.clustering_method = _default_clustering_method

    def _validate_clustering_kws(self):
        if not isinstance(self.clustering_kws, dict):
            warnings.warn("'clustering_kws' must be a dict. "
                          "'clustering_kws' will be converted to dict().", UserWarning)
            self.clustering_kws = dict()

    def _validate_remove_substrings(self):
        if not isinstance(self.remove_substrings, bool):
            warnings.warn("'remove_substrings' must be a boolean. 'remove_substrings' "
                          f"will be set to True", UserWarning)
            self.remove_substrings = True

    def _validate_ranking(self):
        if not (isinstance(self.ranking, str) and self.ranking in ('size', 'bngram')):
            warnings.warn("'ranking' must be one of ('size', 'bngram'). 'ranking' "
                          f"will be converted to 'size'", UserWarning)
            self.ranking = 'size'

    def _check_params(self):
        self._validate_nlp()
        self._validate_n_jobs()
        self._validate_batch_size()
        self._validate_vectorizer_kws()
        self._validate_clustering_method()
        self._validate_clustering_kws()
        self._validate_remove_substrings()
        self._validate_ranking()
