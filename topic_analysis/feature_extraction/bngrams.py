"""
Scikit-compatible module to vectorize text using BNgrams.

References:
    * http://ceur-ws.org/Vol-1150/martin.pdf
    * https://www.researchgate.net/publication/261213425_Mining_Newsworthy_Topics_from_Social_Media
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize, OneHotEncoder
from sklearn.utils.sparsefuncs import min_max_axis, _get_median, csc_median_axis_0
from sklearn.utils.validation import FLOAT_DTYPES
import warnings


TIME_INTERVALS = ('hourly', 'daily', 'weekly', 'monthly', 'quarterly', 'yearly')


class BngramsVectorizer:
    """
    Convert a collection of raw documents to a matrix of temporal document
    frequency-inverse document frequency (`df-idf(t, i)`) features.

    Used for finding novel trends, so if we want to find terms that appear
    more often in one time period than in others. We treat temporal windows
    (i.e. the set of all tweets posted between a start and end time) as
    documents and use them to detect words and phrases that are both new
    and significant.

    We select terms with a high "temporal document frequency-inverse
    document frequency", or `df-idf(t, i)`. The `df-idf(t)` score is
    computed for each term `t` of the current time slot `i` based on
    its document frequency for this time slot and penalized by the
    logarithm of the average of its document frequencies in the previous
    `s` time slots: df−idf(t, i) = (df(t, i) + 1) * idf(t, i) where
    idf(t, i) is computed as 1 / ( log ( (sum(df(t, i-j) for j in range(i, s)) / s ) + 1 ) + 1 ).

    Attributes
    ----------
    vocabulary_ : dict (n_features)
        A mapping of terms to feature indices.

    References
    ----------
    .. [martin2014real] Real-time topic detection with bursty n-grams: RGU's submission to the 2014 SNOW Challenge, 
        Martin, Carlos and Goker, Ayse (2014), CEUR Workshop Proceedings.

    Examples
    --------
    >>> corpus = ['this is the first document',
    ...           'this document is the second document',
    ...           'and this is the third one',
    ...           'is this the first document']
    >>> timestamps = [dt.datetime(2020, 6, 19,14, 51, 38),
    ...               dt.datetime(2020, 6, 17, 14, 14, 24),
    ...               dt.datetime(2020, 6, 21, 14, 16, 9),
    ...               dt.datetime(2020, 6, 21, 14, 44, 39)]
    >>> vocabulary = ['this', 'document', 'first', 'is', 'second', 'the',
    ...               'and', 'one']
    >>> vect = BNgramsVectorizer(vocabulary=vocabulary)
    >>> vect.fit_transform(corpus, timestamps).toarray()
    array([[0.4283029 , 0.4283029 , 0.51597143, 0.4283029 , 0.        ,
            0.4283029 , 0.        , 0.        ],
           [0.4472136 , 0.4472136 , 0.        , 0.4472136 , 0.4472136 ,
            0.4472136 , 0.        , 0.        ],
           [0.45504494, 0.        , 0.        , 0.45504494, 0.        ,
            0.45504494, 0.43520243, 0.43520243],
           [0.48832766, 0.35519127, 0.3980546 , 0.48832766, 0.        ,
            0.48832766, 0.        , 0.        ]])
    >>> vect.transform(corpus, timestamps).shape
    (4, 8)
    """
    def __init__(self, *, encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, dtype=np.float64, norm='l2',
                 time_interval='daily', ma_window_size=2):
        """
        Parameters
        ----------
        ... :
            See documentation [here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

        norm : {'l1', 'l2'}, default='l2'
            Each output row will have unit norm, either:
            * 'l2': Sum of squares of vector elements is 1. The cosine
            similarity between two vectors is their dot product when l2 norm has
            been applied.
            * 'l1': Sum of absolute values of vector elements is 1.
            See :func:`preprocessing.normalize`

        time_interval : {'hourly', 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'}, default='daily'
            Time interval over which to compute the temporal document frequency.

        ma_window_size : int, default=2
            Length of previous time slots to use to calculate the inverse document
            frequencies
        """

        self.dtype = dtype
        self._validate_dtype()
        self.norm = norm
        self.time_interval = time_interval.lower()
        self.ma_window_size = ma_window_size
        self._vectorizer = CountVectorizer(
            input='content', encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=True,
            dtype=self.dtype)

    def transform(self, raw_documents, timestamps):
        self._check_params()
        raw_documents = np.array(raw_documents, dtype=np.object)

        # create time slots
        timestamps = pd.to_datetime(pd.Series(timestamps))\
            .dt.to_period(self.time_interval[0].upper())
        ohe = OneHotEncoder(sparse=True)
        dt = ohe.fit_transform(np.array([timestamps]).T)
        time_slots = ohe.categories_[0]
        del ohe
        S = time_slots.shape[0]

        # fit vocabulary
        if self._vectorizer.vocabulary is None:
            self._vectorizer.vocabulary =\
                self._vectorizer.fit(raw_documents).vocabulary_
        V = len(self._vectorizer.vocabulary)

        # get document frequencies per time slot `i`
        # lil_matrix are better for slicing which we do in the loop
        # then convert to csr matrix for matrix multiplications
        tf = sp.lil_matrix((raw_documents.shape[0], V),
                           dtype=self.dtype)
        df = np.zeros((S, V), dtype=self.dtype)
        for i, ts in enumerate(time_slots):
            indices = np.where(timestamps == ts)[0]
            tf[indices] = clone(self._vectorizer)\
                .fit_transform(raw_documents[indices])
            df[i] = tf[indices].sum(0)
        tf = tf.tocsr()

        # get temporal document frequency-inverse document frequency
        df_idf = pd.DataFrame(df).shift(1).fillna(0.)\
            .rolling(window=self.ma_window_size, min_periods=1, axis=0)\
            .mean().values
#         display('df', df)
#         display('df_idf_rolling_sum', pd.DataFrame(df).shift(1).fillna(0.)\
#             .rolling(window=self.ma_window_size, min_periods=1, axis=0)\
#             .sum().values)
#         display('df_idf_rolling_mean', df_idf)
        df_idf = (df + 1.) / (np.log(df_idf + 1.) + 1.)
#         display('df_idf', df_idf)
#         display('dt * df_idf', dt * df_idf)

        # get document-term matrix
        X = tf.multiply(sp.csr_matrix(dt * df_idf))

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X

    def fit_transform(self, raw_documents, timestamps, y=None):
        """
        Learn global vocabulary over all documents and
        `df-idf(t, i)` from training set.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode
            or file objects.

        timestamps : iterable
            timestamp of the documents. An iterable which
            yields datetime objects.

        y : None
            This parameter is not needed to compute BNgrams.

        Returns
        -------
        self : object
            Fitted vectorizer.
        """
        return self.transform(raw_documents, timestamps)

    @property
    def vocabulary_(self):
        return self._vectorizer.vocabulary_

    def get_feature_names(self):
        return self._vectorizer.get_feature_names()

    def _validate_dtype(self):
        if self.dtype not in FLOAT_DTYPES:
            warnings.warn("Only {} 'dtype' should be used. {} 'dtype' will "
                          "be converted to np.float64."
                          .format(FLOAT_DTYPES, self.dtype), UserWarning)
            self.dtype = np.float64

    def _validate_time_interval(self):
        if self.time_interval not in TIME_INTERVALS:
            warnings.warn("Only {} 'time_interval' should be used. {} "
                          "'time_interval' will be converted to 'daily'."
                          .format(TIME_INTERVALS, self.time_interval), UserWarning)
            self.time_interval = 'daily'

    def _validate_ma_window_size(self):
        if not (isinstance(self.ma_window_size, int)
                and self.ma_window_size > 0):
            warnings.warn("'ma_window_size' must be a positive integer", UserWarning)
            self.ma_window_size = 2

    def _check_params(self):
        self._validate_dtype()
        self._validate_time_interval()
        self._validate_ma_window_size()
