"""
Fighting Words.

GitHub: https://github.com/jmhessel/FightingWords/blob/master/fighting_words_py3.py
Paper: http://languagelog.ldc.upenn.edu/myl/Monroe.pdf
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer as CV
import string
from topic_analysis.utils import _logging


# Global variables
exclude = set(string.punctuation)
logger = _logging.get_logger()


def basic_sanitize(in_string : str):
    """Returns a very roughly sanitized version of the input string."""  
    in_string = ''.join([ch for ch in in_string if ch not in exclude])
    in_string = in_string.lower()
    in_string = ' '.join(in_string.split())
    return in_string


def bayes_compare_language(l1, l2, ngram_range=(1, 3), prior=.01, cv=None, counts_mat=None):
    """
    Parameters
    ----------
    l1, l2 : Iterable[str]
        list of strings from each language sample

    ngram_range : Tuple[int, int], default=(1,3)
        an integer describing up to what n-gram you want to consider
        (1 is unigrams, 2 is bigrams + unigrams, etc). Ignored if a
        custom CountVectorizer is passed.

    prior : Union[float, array[float]]
        a float describing a uniform prior, or a vector describing a
        prior over vocabulary items. If you're using a predefined
        vocabulary, make sure to specify that when you make your
        CountVectorizer object.

    cv : Optional[sklearn.feature_extraction.text.CountVectorizer], default=None
        Pass this if you have pre-defined vocabulary. If None, by
        default an sklearn CV with min_df=10, max_df=.5, and
        ngram_range=(1,3) with max 15000 features.

    counts_mat : Optional[np.ndarray[len(l1 + l2), k]], default=None
        Counts matrix with size equal to length of `l1 + l2` (must
        also be in that order) and with k features. Pass this if
        you already have a dataset vectorized. If given, then the
        vectorizer must also be passed to `cv`.

    Returns
    -------
    z_scores : pd.DataFrame
        A pandas DataFrame of shape (|Vocab|, 2) with (n-gram, z-score) pairs.

    array[array[float]]:
        A 2-row matrix of counts of terms in l1 and l2 respectively.
    """
    if cv is None and type(prior) is not float:
        raise ValueError("If using a non-uniform prior, please also pass a count "
                         "vectorizer with the vocabulary parameter set.")
    if counts_mat is not None:
        assert isinstance(cv, CV)

    # clean the text
    if counts_mat is None:
        logger.info('Basic cleaning of the text')
        l1 = [basic_sanitize(l) for l in l1]
        l2 = [basic_sanitize(l) for l in l2]

    # initialize count vectorizer
    if counts_mat is None:
        logger.info('Vectorizing documents with CountVectorizer')
        if cv is None:
            cv = CV(decode_error='ignore', min_df=10, max_df=.5,
                    ngram_range=ngram_range, binary=False,
                    max_features=15000)
        counts_mat = cv.fit_transform(l1 + l2).toarray()
    vocab_size = len(cv.vocabulary_)
    logger.info("Vocab size is {}".format(vocab_size))

    # Now sum over languages...
    if type(prior) is float:
        priors = np.array([prior for i in range(vocab_size)])
    else:
        priors = prior
    z_scores = np.empty(priors.shape[0])
    count_matrix = np.empty([2, vocab_size], dtype=np.float32)
    count_matrix[0, :] = np.sum(counts_mat[:len(l1), :], axis=0)
    count_matrix[1, :] = np.sum(counts_mat[len(l1):, :], axis=0)
    a0 = np.sum(priors)
    n1 = 1. * np.sum(count_matrix[0,:])
    n2 = 1. * np.sum(count_matrix[1,:])

    logger.info("Comparing language...")
    for i in range(vocab_size):
        # compute delta
        term1 = np.log((count_matrix[0, i] + priors[i]) / (n1 + a0 - count_matrix[0, i] - priors[i]))
        term2 = np.log((count_matrix[1, i] + priors[i]) / (n2 + a0 - count_matrix[1, i] - priors[i]))        
        delta = term1 - term2

        # compute variance on delta
        var = 1. / (count_matrix[0, i] + priors[i]) + 1. / (count_matrix[1, i] + priors[i])

        # store final score
        z_scores[i] = delta / np.sqrt(var)

    index_to_term = {v:k for k, v in cv.vocabulary_.items()}
    sorted_indices = np.argsort(z_scores)
    z_scores = pd.DataFrame([(index_to_term[i], z_scores[i]) for i in sorted_indices], columns=['term', 'z-score'])
    logger.info("Done")

    return z_scores, count_matrix


def plot_fighting_words(
        n_gram_z_scores: pd.DataFrame, count_matrix: np.ndarray,
        sig_val=1.96, max_label_size=15, **plot_kws):
    """
    Plot fighting words.
    Example: https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/convokit/fighting_words/demos/fightingwords_demo.ipynb

    Parameters
    ----------
    z_scores : pd.DataFrame
        A pandas DataFrame of shape (|Vocab|, 2) with (n-gram, z-score) pairs.

    array[array[float]]:
        A 2-row matrix of counts of terms in l1 and l2 respectively.

    sig_val : float, default=1.96
        Significance value. Any term with z-score > sig_val or < -sig_val
        will be annotated and colored in the plot. The default 1.96 means
        95% confidence interval on z-score. You can also use 2.33 for 99%.

    max_label_size: int, default=15
        For the text labels (annotations), set the largest possible size
        for any text label (the rest will be scaled accordingly).

    plot_kws : kwargs
        Keywords arguments for plt.subplot()

    Returns
    -------
    fig, ax: matplotlib objects
    """
    x_vals = count_matrix.sum(axis=0)
    y_vals = list(n_gram_z_scores['z-score'])
    sizes = abs(np.array(y_vals))
    scale_factor = max_label_size / max(sizes)
    sizes *= scale_factor
    neg_color, pos_color, insig_color = ('orange', 'red', 'grey')
    colors = []
    annots = []
    terms = list(n_gram_z_scores['term'])

    for i, y in enumerate(y_vals):
        if y > sig_val:
            colors.append(pos_color)
            annots.append(terms[i])
        elif y < -sig_val:
            colors.append(neg_color)
            annots.append(terms[i])
        else:
            colors.append(insig_color)
            annots.append(None)

    fig, ax = plt.subplots(**plot_kws)
    ax.scatter(x_vals, y_vals, c=colors, s=sizes, linewidth=0)
    for i, annot in enumerate(annots):
        if annot is not None:
            ax.annotate(annot, (x_vals[i], y_vals[i]), color=colors[i], size=sizes[i])

#     ax.legend()
    ax.set_xscale('log')
    ax.set_title("Weighted log-odds ratio vs. Frequency of word within class")

    return fig, ax
