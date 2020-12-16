from .bngrams import BngramsVectorizer
from .noun_phrases import NounPhraseMatcher
import numpy as np
import pandas as pd
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.sparsefuncs import min_max_axis, _get_median, csc_median_axis_0
from topic_analysis.utils import _logging
import warnings


# Global variables
_default_n_jobs = 1
_default_batch_size = 1000
logger = _logging.get_logger()


def get_ranked_phrases(
        nlp, raw_documents, timestamps=None, *, include_verb_phrases=False,
        minlen=1, maxlen=8, n_jobs=_default_n_jobs, batch_size=_default_batch_size,
        stop_phrases=[], vectorizer='bngram', aggfunc='sum', **vectorizer_kws):
    """
    Get phrases ranked by either TF-IDF (importance) score or BNgram (novelty) score.

    Parameters
    ----------
    nlp : spacy.language.Language
        Spacy language model

    raw_documents : Iterable[str]
        An iterable which yields either str objects.

    timestamps : Iterable[str]
        timestamp of the documents. An iterable which
        yields datetime objects. Only used when
        `vectorizer='bngram'`.

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

    aggfunc : Union[str, callable, NoneType], default='sum'
        Function to aggregate over the scores per document
        for a single phrase to rank. One of ('sum', 'mean',
        'max', 'median', 'median_ignore_0', callable that
        accepts sparse matrix, None). If None, this function
        will return the vectorized documents and the vectorizer
        directly.

    vectorizer_kws : dict
        Keyword arguments for TfidfVectorizer

    Returns
    -------
    ranked_phrases : Union[pandas.DataFrame, Tuple[array[N, M], vectorizer]]
        If aggfunc is not None, returns the dataframe with the extracted
        n-gram / phrase and sorted descending by the aggregated bngram /
        td-idf scores, else returns the vectorized documents (where
        N=len(raw_documents) and M=len(phrases)) and the vectorizer object,
    """
    assert vectorizer in ('bngram', 'tfidf')
    stop_phrases = set(stop_phrases)

    # get candidate phrases
    nlp.add_pipe(NounPhraseMatcher(
        lowercase=True, lemmatize=True,
        include_verb_phrases=include_verb_phrases,
        minlen=minlen, maxlen=maxlen))

    # extract phrases
    def process_chunk(texts):
        return list(nlp.pipe(texts))

    logger.info('Tokenizing, tagging and extracting noun phrases '
                'per documents with spacy')
    n_jobs = psutil.cpu_count(logical=False)\
        if n_jobs == -1 else n_jobs
    raw_documents = list(nlp.pipe(
        raw_documents, batch_size=batch_size, n_process=n_jobs))

    # vectorize the texts
    if 'norm' in vectorizer_kws and aggfunc is not None:
        warnings.warn("'vectorizer_kws' should not contain 'norm'. "
                      "'vectorizer_kws['norm']' will be replaced.", UserWarning)
        vectorizer_kws['norm'] = None
    if 'analyzer' in vectorizer_kws:
        warnings.warn("'vectorizer_kws' should not contain 'analyzer'. "
                      "'vectorizer_kws['analyzer']' will be replaced.", UserWarning)
    vectorizer_kws['analyzer'] = lambda doc: [p for p in doc._.noun_phrases if p not in stop_phrases]
    if vectorizer == 'bngram':
        if timestamps is None:
            raise ValueError('Parameter `timestamps` cannot be None if `vectorizer=bngram`.')
        vectorizer = BngramsVectorizer(**vectorizer_kws)
        logger.info('Vectorizing documents with BNgrams')
        X = vectorizer.fit_transform(raw_documents, timestamps)
    elif vectorizer == 'tfidf':
        vectorizer = TfidfVectorizer(**vectorizer_kws)
        logger.info('Vectorizing documents with TF-IDF')
        X = vectorizer.fit_transform(raw_documents)
    else:
        raise ValueError(f'Unknown vectorizer={vectorizer} given.')

    logger.info('Scoring phrases')
    if aggfunc == 'sum':
        scores = np.array(X.tocsc().sum(0))[0]
    elif aggfunc == 'mean':
        scores = np.array(X.tocsc().mean(0))[0]
    elif aggfunc == 'max':
        scores = min_max_axis(X.tocsc(), axis=0, ignore_nan=True)[1]
    elif aggfunc == 'median':
        scores = csc_median_axis_0(X.tocsc())
    elif aggfunc == 'median_ignore_0':
        scores = _get_median(X.tocsc(), 0)
    elif callable(aggfunc):
        scores = aggfunc(X.tocsc())
    elif aggfunc is None:
        return X, vectorizer
    else:
        raise ValueError(f'Unknown method: {aggfunc}')

    logger.info('Rank phrases based on score')
    ranked_phrases = pd.DataFrame(
        list(zip(vectorizer.get_feature_names(), scores)),
        columns=['phrase', 'score'])
    ranked_phrases = ranked_phrases\
        .sort_values('score', ascending=False)\
        .reset_index(drop=True)

    return ranked_phrases
