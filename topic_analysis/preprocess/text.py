import numpy as np
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span
from spacy.util import filter_spans
import string
from symspellpy import SymSpell, Verbosity
from topic_analysis import SUPPORTED_LANGUAGES
from topic_analysis.feature_extraction import NounPhraseMatcher
from topic_analysis.utils import _joblib as jl
from topic_analysis.preprocess.utils import get_component_name
from typing import Callable, List
from unidecode import unidecode

try:
    # import pystemmer by default because its fast (C-based)
    import Stemmer
    _pystemmer = True
except ImportError:
    from nltk.stem.snowball import SnowballStemmer
    _pystemmer = False


class SpacyTokenizer:
    """
    Tokenize string with spacy
    """

    name = 'spacy_tokenizer'

    def __init__(self, nlp):
        """
        Parameters
        ----------
        nlp : spacy.language.Language
            Spacy language object
        """
        self.tokenizer = nlp.tokenizer
        assert self.tokenizer is not None

    def __call__(self, doc : str):
        """
        Parameters
        ----------
        doc : str
            Input string

        Returns
        -------
        tokens : List[str]
            List of tokens
        """
        return [t.text for t in self.tokenizer(str(doc))]


class RegexCleaner:
    """
    Replace patterns in string with a single-space. Predefined patterns
    include:
        1. BertOOVPunctuation: [, ], {, }, #, @, $, %, *, +, \, <, =, >, ^, _, |, ~
        2. Dates: e.g., dd/mm/yy, dd-mm-yy, dd/mm/19yy, dd/mm/20yy, dd-mm-yyyy
        3. Email: e.g., abc.xyz@def.com
        4. english_contractions: e.g., 's, 't, 'll, 'll've, etc.
        5. LargeIntegers: >=7 digit numbers which are not part of a word
        6. Linebreak: e.g., \n, \r\n, \\r\\n, \r\n\n etc.
        7. NL_pc6: e.g., 5000AB, 4000 AB, 3000 ab, 2000ab etc.
        8. NL_PhoneNumber: 
        9. Numbers: e.g., 1.25, 2, 20 20.8 etc.
        10. URL: anything that starts with `http` followed by non-space characters
    """

    name = 'regex_cleaner'
    WholeWordOnly = lambda w: r'\b{}\b'.format(w)
    BertOOVPunctuation = r'[\[\]{}#@$%*+\<=>^_|~]'
    Dates = WholeWordOnly(r'\d{1,2}([-/])\d{1,2}\1(?:(19|20)\d{2}|\d{2})')
    Email = WholeWordOnly(r"[a-zA-Z0-9_.+-]+[@][a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    english_contractions = r"(?<=[A-Za-z])['](s|t[']ve|d[']ve|ll[']ve|d[']y|ve|t|d|ll|m|am|re|alls?|cause)"
    LargeIntegers = WholeWordOnly(r'\d{7,}')
    Linebreak = r'(?:\r|\\r)?(?:\n|\\n)+'
    NL_pc6 = WholeWordOnly(r'[0-9]{4}\s?[a-zA-Z]{2}')
    NL_PhoneNumber = r'((\+?31)|(0031)|0)(\(0\)|)(\d{1,3})(\s|\-|)(\d{8}|\d{4}\s\d{4}|\d{2}\s\d{2}\s\d{2}\s\d{2})'
    Numbers = WholeWordOnly(r'\d+(?:[.,]\d+)?')
    URL = r'http\S+'

    def __init__(
            self, remove_linebreaks=False, remove_email=False,
            remove_urls=False, remove_dates=False,
            remove_NL_postcodes=False, remove_NL_phonenum=False,
            replace_english_contractions=False,
            remove_punctuations=False, remove_numbers=False,
            remove_double_quotes=False, remove_large_integers=False,
            additional_regexes=[]):
        """
        Parameters
        ----------
        remove_* : bool
            Only if true, corresponding regex will be applied.

        additional_regexes : Optional[List[str]]
            Additional regexes can be supplied to clean.
        """
        regex_list = []
        if remove_linebreaks:
            regex_list.append(self.Linebreak)
        if remove_email:
            regex_list.append(self.Email)
        if remove_urls:
            regex_list.append(self.URL)
        if remove_dates:
            regex_list.append(self.Dates)
        if remove_NL_postcodes:
            regex_list.append(self.NL_pc6)
        if remove_NL_phonenum:
            regex_list.append(self.NL_PhoneNumber)
        if replace_english_contractions:
            regex_list.append(english_contractions)
        if remove_punctuations:
            regex_list.append(self.BertOOVPunctuation)
        if remove_numbers:
            regex_list.append(self.Numbers)
        if remove_double_quotes:
            regex_list.append(r'["]')
        if remove_large_integers:
            regex_list.append(self.LargeIntegers)
        regex_list.extend(additional_regexes)
        self.compiled_regex_list = re.compile(r"|".join(regex_list))

    def __call__(self, doc: str):
        """
        Parameters
        ----------
        doc : str
            Input string

        Returns
        -------
        doc : str
            String with regex patterns removed
        """
        return self.compiled_regex_list.sub(' ', doc)


class Replace3Plus:
    """
    Replace any non-space character repeated consecutively 3+ times
    with max 2 times the same character.
    """

    name = 'replace_3_plus'
    ThreePlusRepeatedAny = re.compile(r'(?:(\S)\1{2,})')

    def __call__(self, doc: str):
        """
        Parameters
        ----------
        doc : str
            Input string

        Returns
        -------
        doc : str
            String with 3+ repeated characters replaced
        """
        return self.ThreePlusRepeatedAny.sub(r'\1\1', doc)


class SplitSpecialCharacters:
    """
    Add space to both sides of special characters. Special
    characters in numbers are like decimal points are
    avoided.
    """

    name = 'split_special_characters'
    SpecialCharacters = re.compile(r"((?<!\d)[^A-Za-z0-9\s']+(?=\d)|(?<=\d)[^A-Za-z0-9\s']+(?!\d)|(?<!\d)[^A-Za-z0-9\s']+(?!(?<=[.])(?:[A-Za-z][.]|\s|$))(?!\d)|(?<=\d)[^\w,.\s](?=\d))")

    def __call__(self, doc: str):
        """
        Parameters
        ----------
        doc : str
            Input string

        Returns
        -------
        doc : str
            String with special characters separated
        """
        return self.SpecialCharacters.sub(r' \1 ', doc)


class RemoveLineTerminators:
    """
    Remove the `., ?, !` at the end of a string.
    """

    name = 'remove_line_terminators'
    LineTerminator = re.compile(r'[.?!]\s*$')

    def __call__(self, doc: str):
        """
        Parameters
        ----------
        doc : str
            Input string

        Returns
        -------
        doc : str
            String with line terminators removed
        """
        return self.LineTerminator.sub('', doc)


class FixSpaces:
    """
    Replace 2+ consecutive spaces in a string with a
    single space character.
    """
    name = 'fix_spaces'
    MultipleSpaces = re.compile(r'\s{2,}')

    def __call__(self, doc: str):
        """
        Parameters
        ----------
        doc : str
            Input string

        Returns
        -------
        doc : str
            String with spaces fixed
        """
        return self.MultipleSpaces.sub(' ', doc).strip()


class SpellChecker:
    """
    Spell check a string with max edit distance of 2. Only
    applied on alphabet character words (not alphanumeric).
    Spell check is applied independently to each token. Token
    of size <=2 are skipped.
    """

    name = 'spell_checker'

    def __init__(self, vocab_files_path: List[str], tokenizer=str.split):
        """
        Parameters
        ----------
        vocab_files_path : List[str]
            List of file paths to vocabulary used by the spell
            checker for correction. Vocabulary files must be
            one token per line and the tokens should be in the
            first column left-to-right if there are multiple
            columns.

        tokenizer : Callable[str, List[str]], default=str.split
            Takes as input a string, outputs a list of tokens.
        """
        self.tokenizer = tokenizer
        self.spellchecker = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        for fp in vocab_files_path:
            assert self.spellchecker.load_dictionary(fp, 0, 1)

#         # https://stackoverflow.com/questions/1528932/how-to-create-inline-objects-with-properties
#         self.spellchecker.update({
#             'dummy': type('_', (object,), dict(lookup=lambda w, *_, **__: w))()
#         })

    def _correct(self, token: str) -> str:
        """
        Parameters
        ----------
        token : str
            Input string

        Returns
        -------
        corrected : str
            corrected token
        """
        # https://github.com/mammothb/symspellpy/issues/7
        o = self.spellchecker\
            .lookup(
                token, verbosity=Verbosity.TOP,
                max_edit_distance=2,
                ignore_token=r'\w{,2}',  # ignore tokens of size 2 or less
                transfer_casing=True)
        if not o: return token

        word = o[0].term
        if token[0].isupper():
            word = word[0].upper() + word[1:]

        # find start punctuation
        start_idx = 0
        start_punct = ''
        while token[start_idx] in string.punctuation:
            start_punct += token[start_idx]
            if start_idx + 1 < len(token):
                start_idx += 1
            else:
                break

        # find end punctuation
        end_idx = 1
        end_punct = ''
        while token[-end_idx] in string.punctuation:
            end_punct += token[-end_idx]
            if end_idx - 1 > 0:
                end_idx -= 1
            else:
                break

        return start_punct + word + end_punct

    def __call__(self, doc: str) -> str:
        """
        Parameters
        ----------
        doc : str
            Input string

        Returns
        -------
        doc : str
            spell checked string
        """
        return " ".join([self._correct(w) if w.isalpha() else w for w in self.tokenizer(doc)])


class SplitWordCompounds(BaseEstimator, TransformerMixin):
    """
    Split word compounds in a string.
    e.g., internetverbinding -> internet verbinding
    """

    name = 'split_word_compounds'

    def __init__(
            self, tokenizer: Callable[[str], List[str]] = str.split,
            min_df: int = 1, bigram_unigram_prob: float = .001):
        """
        Parameters
        ----------
        tokenizer : Callable[List[str], List[str]], default=str.split
            Takes as input a string, outputs a list of tokens.

        min_df : Union[int, float], default=1
            When building the replacements vocabulary ignore unigrams
            that have a document frequency strictly lower than the
            given threshold. It can also be the percentage of total
            documents in the range (0.0, 1.0).

        bigram_unigram_prob : float, default=.001
            Minimum ratio of bigram to unigram version of the same tokens
            for the unigram version to be split into the bigram version.
            e.g., count[('inter', 'net')] / (count[('inter', 'net')] + count['internet'])
            must be >= `bigram_unigram_prob` for 'internet' to be split into
            'inter' and 'net'.
        """
        self.tokenizer = tokenizer
        self.min_df = min_df
        self.bigram_unigram_prob = bigram_unigram_prob

    def check_params(self):
        if not callable(self.tokenizer):
            raise ValueError(f'`tokenizer={self.tokenizer}` must be a callable.')
        if not (isinstance(self.min_df, int) and self.min_df > 0)\
        and not (isinstance(self.min_df, float) and 0. < self.min_df < 1.):
            raise ValueError(f"`min_df={self.min_df}` must be a positive "
                             "integer >= 1 or a float between 0.0 and 1.0")
        if not 0 < self.bigram_unigram_prob < 1:
            raise ValueError(f'`bigram_unigram_prob={self.bigram_unigram_prob}` '
                             'must be in range (0, 1).')

    def fit(self, X, y=None):
        """
        Scan through all texts to build a dictionary of
        word compounds

        Parameters
        ----------
        X : Iterable[str]
            Iterable of strings

        y : NoneType
            Only there for scikit-learn compatibility

        Returns
        -------
        self : SplitWordCompounds
        """
        self.check_params()
        X = check_array(X, dtype=np.str_, ensure_2d=False)
        min_df = int(self.min_df * X.shape[0])\
            if isinstance(self.min_df, float)\
            else self.min_df

        # tokenize
        tokenizer = np.vectorize(self.tokenizer, otypes=[np.object])
        X = tokenizer(np.char.lower(X))

        # build dictionary of unibigram replacements
        from collections import Counter
        unigrams = Counter([w for e in X for w in e])
        unigrams = {
            k: v for k, v in unigrams.items()
            if v >= min_df and not k.isnumeric()}
        bigrams = Counter([
            w for e in X for w in zip(e, e[1:])
            if not (w[0].isnumeric() or w[1].isnumeric())])
        unibigrams = {
            w[0] + w[1]: w for w in bigrams
            if bigrams[w] / (bigrams[w] + unigrams.get(w[0] + w[1], np.inf)) >= self.bigram_unigram_prob}
        self._replacements = {
            k: len(v[0]) for k, v in unibigrams.items()
            if k in unigrams}

        return self

    def transform(self, X, *_):
        """
        Replace word compounds with split form.

        Parameters
        ----------
        X : Iterable[str]
            Iterable of strings

        Returns
        -------
        X : Iterable[str]
            Same as input with word compounds fixed
        """
        check_is_fitted(self, '_replacements')
        X = check_array(X, dtype=np.str_, ensure_2d=False)

        # apply replacements
        if self._replacements:
            replace_regex = re.compile(r'\b(%s)\b' % '|'.join(
                [re.escape(k) for k in self._replacements.keys()]), re.IGNORECASE)
            def do_replace(match):
                text = match.group(0)
                idx = self._replacements.get(text.lower(), None)
                if not idx:
                    return text
                return text[:idx] + ' ' + text[idx:]
            replace = np.vectorize(lambda text: replace_regex.sub(do_replace, text))
            X = replace(X)
        
        return X


class StopwordsRemover:
    """
    Given a set of stopwords, remove them from
    a string after splitting on space character.
    """

    name = 'stopwords_remover'

    def __init__(self, stopwords):
        """
        Parameters
        ----------
        stopwords : Iterable[str]
            List of stopwords
        """
        self.stopwords = set(stopwords)

    def __call__(self, doc : str):
        """
        Parameters
        ----------
        doc : str
            Input string

        Returns
        -------
        doc : str
            String with stopwords removed
        """
        return ' '.join([t for t in doc.split() if t not in self.stopwords])


# class POSPhraseMatcher:
#     """
#     Extract phrases using spaCy's rule-based matching on
#     part-of-speech (POS) tags. Predefined patterns include:
#         1. noun phrases
#         2. noun-verb phrases
#         3. verb phrases
#     """

#     name = 'pos_phrases_matcher'
#     patterns = {
#         'noun_phrases': [
#             {'POS': 'ADJ', 'OP': '*'},
#             {'POS': {'REGEX': '(PROPN|NOUN)'}, 'OP': '+'}],

#         'noun_verb_phrases': [
#             {'POS': 'ADJ', 'OP': '*'},
#             {'POS': {'REGEX': '(PROPN|NOUN)'}, 'OP': '+'},
#             {'POS': {'REGEX': '(CONJ|CCONJ|ADP)'}, 'OP': '?'},
#             {'POS': 'VERB', 'OP': '*'}],

#         'verb_phrases': [
#             {'POS': 'ADV', 'OP': '*'},
#             {'POS': 'ADJ', 'OP': '*'},
#             {'POS': 'VERB', 'OP': '+'}],
#     }

#     def __init__(
#             self, nlp, additional_patterns=dict(), normalize=False,
#             attr_name='phrases', force_extension=True):
#         """
#         Parameters
#         ----------
#         nlp: spacy.language.Language
#             Spacy language object

#         additional_patterns: Dict[str, List[Dict[str, str]]], default={}
#             any additional patterns to extract

#         normalize: bool, default=False
#             Stem and lowercase the text before pattern matching

#         attr_name: str, default='phrases'
#             Doc._ extension attribute name to store the phrases

#         force_extension: bool, default=True
#             Overwrite existing Doc._ extension with same name as
#             `attr_name`.
#         """

#         for p in additional_patterns:
#             if 'POS' not in p:
#                 raise ValueError(f'Pattern {p} must contain POS attribute.')
#         self.patterns.update(additional_patterns)
#         self.normalize = normalize

#         # build matcher for getting phrases
#         self.matcher = Matcher(nlp.vocab)
#         for k, v in self.patterns.items():
#             self.matcher.add(k, None, v)

#         # set document extension
#         self.attr_name = attr_name
#         Doc.set_extension(self.attr_name, default=[], force=force_extension)

#     def __call__(self, doc: Doc):
#         """
#         Parameters
#         ----------
#         doc : spacy.tokens.Doc
#             Input spacy doc object

#         Returns
#         -------
#         doc : spacy.tokens.Doc
#             Output spacy doc object with '_.{attr_name}' attribute set.
#         """

#         def process(text: Span):
#             if self.normalize:
#                 text = text.lemma_
#             text = str(text)
#             if self.normalize:
#                 text = text.lower()
#             return text

#         phrases = set([
#             process(p) for p in filter_spans(
#             [doc[start:end] for _, start, end in self.matcher(doc)])])
#         doc._.set(self.attr_name, phrases)

#         return doc


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocess text data.
    """

    def __init__(
            self, nlp, fix_non_ascii=True, lowercase=False, regex_clean_kwargs={},
            replace_3plus_repeated=False, spell_check_kwargs={},
            fix_compounds=False, fix_compounds_kwargs={}, remove_stopwords=False,
            additional_stopwords=[], stem=False, lemmatize=False, pos_parse=False,
            pos_parse_kwargs={}, remove_line_terminators=False, n_jobs=1,
            batch_size='auto'):
        """
        Parameters
        ----------
        nlp : spacy.language.Language
            Spacy language object

        fix_non_ascii : bool, default=True
            Replace unicode characters with ascii equivalent
            wherever possible.

        lowercase: bool, default=False
            Lowercase the text.

        regex_clean_kwargs : dict, default={}
            Arguments for `RegexCleaner`. If empty list, then
            no regex cleaning is applied.

        replace_3plus_repeated : bool, default=False
            Apply `Replace3Plus`.

        spell_check_kwargs : default={}
            If not empty, apply `SpellChecker(**spell_check_kwargs)`.

        fix_compounds : bool, default=False
            Apply `SplitWordCompounds`.

        fix_compounds_kwargs : dict, default={}
            Arguments for `SplitWordCompounds`.

        remove_stopwords : bool, default=False
            Apply `StopwordsRemover`.

        additional_stopwords : Iterable[str]
            Iterable of additional stopwords.

        stem : bool, default=False
            If `stem=True and lemmatize=False`, stem text using
            `nltk.stem.snowball.SnowballStemmer`. Note that stemming
            is always more aggresive than lemmatization in converting
            a word to its root form.

        lemmatize : bool, default=False
            Apply spacy lemmatization.

        pos_parse: bool, default=False
            Convert input text into a bag-of-phrases.
            Output will still be a string where each phrase
            will be `_` delimited and separated by space
            from other phrases, e.g., 'A_b c_d_e c_d'.

        pos_parse_kwargs : dict, default={}
            Arguments for `NounPhraseMatcher`.

        remove_line_terminators : bool, default=False
            Apply `RemoveLineTerminators`.

        n_jobs : int, default=1
            Number of processes to parallelize the task.
             if n_jobs <= -1, Use all cores
            elif n_jobs == 0, No parallelization
                        else, Use minimum of number of cores available
                              and `n_jobs`

        batch_size : Union[int, str], default='auto'
            Positive integer or 'auto'. Used fo chunking data
            into batches for parallelization.
        """
        # global _swc_transform, stemmer_stem, do_lemmatize, extract_phrases
        assert nlp.lang in SUPPORTED_LANGUAGES
        assert (isinstance(batch_size, int) and batch_size > 0) or batch_size == 'auto'

        self.pipeline : List[[str, callable]] = []
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        tokenizer = SpacyTokenizer(nlp)

        # build pipeline
        if fix_non_ascii:
            self.add_pipe(unidecode, name='fix_non_ascii')
        if lowercase:
            self.add_pipe(str.lower, name='lowercase')
        if regex_clean_kwargs:
            self.add_pipe(RegexCleaner(**regex_clean_kwargs))
        if replace_3plus_repeated:
            self.add_pipe(Replace3Plus())
        self.add_pipe(SplitSpecialCharacters())
        if spell_check_kwargs:
            if 'tokenizer' not in spell_check_kwargs:
                spell_check_kwargs['tokenizer'] = tokenizer
            self.add_pipe(SpellChecker(**spell_check_kwargs))
        if fix_compounds:
            if 'tokenizer' not in fix_compounds_kwargs:
                fix_compounds_kwargs['tokenizer'] = tokenizer
            self._swc = SplitWordCompounds(**fix_compounds_kwargs)
            def _swc_transform(doc):
                return str(self._swc.transform([doc])[0])
            self.add_pipe(_swc_transform, name=self._swc.name)
        if remove_stopwords:
            sw = nlp.Defaults.stop_words
            sw |= set(additional_stopwords)
            self.add_pipe(StopwordsRemover(sw))
        if stem:
            if _pystemmer:
                stemmer = Stemmer.Stemmer(SUPPORTED_LANGUAGES[nlp.lang])
                def stemmer_stem(doc):
                    return ' '.join(stemmer.stemWords(doc.split()))
            else:
                stemmer = SnowballStemmer(SUPPORTED_LANGUAGES[nlp.lang])
                def stemmer_stem(doc):
                    return ' '.join([stemmer.stem(t) for t in doc.split()])
            self.add_pipe(stemmer_stem, name='stem')
        elif lemmatize:
            self.add_pipe(nlp, name='make_spacy_doc_1')

            def do_lemmatize(doc):
                words = []
                for t in doc:
                    word = t.lemma_
                    # transfer casing
                    if t.text[0].isupper():
                        word = word[0].upper() + word[1:]
                    words.append(word)
                return ' '.join(words)

            self.add_pipe(do_lemmatize, name='lemmatize')
        if pos_parse:
            pos_matcher = NounPhraseMatcher(**pos_parse_kwargs)
            self.add_pipe(nlp, name='make_spacy_doc_2')
            def extract_phrases(doc):
                doc = pos_matcher(doc)
                # replace spaces in phrases with underscore and join
                # them with space into a single string
                return ' '.join([
                    t.replace(' ', '_')
                    for t in getattr(doc._, pos_matcher.attr_name, [])])
            self.add_pipe(extract_phrases, name=pos_matcher.name)
        if remove_line_terminators:
            self.add_pipe(RemoveLineTerminators())
        self.add_pipe(FixSpaces())

    @property
    def pipe_names(self):
        """List of names of functions in the pipeline"""
        return [name for name, _ in self.pipeline]

    def add_pipe(
            self, component, name=None, before=None, after=None,
            first=None, last=None):
        """
        Add a callable component (function) to the pipeline that
        takes a string as input and gives back a string as output.

        Parameters
        ----------
        component : callable
            The component to call in the pipeline.

        name : str
            The name of the component. Must be unique in the pipeline.

        before : str
            The name of the component to put before in the pipeline.

        after : str
            The name of the component to put after in the pipeline.

        first : bool
            Should be first element?

        last : bool
            Should be last element?
        """
        if not hasattr(component, "__call__"):
            msg = "Not a valid pipeline component. Expected callable, but "
            "got {} (name: '{}')."
            raise ValueError(msg.format(component, name))
        if name is None:
            name = get_component_name(component)
        if name in self.pipe_names:
            msg = "'{}' already exists in pipeline. Existing names: {}"
            raise ValueError(msg.format(name, self.pipe_names))
        if sum([bool(before), bool(after), bool(first), bool(last)]) >= 2:
            msg = "Invalid constraints. You can only set one of the following: "
            "before, after, first, last."
            raise ValueError(msg)
        pipe_index = 0
        pipe = (name, component)
        if last or not any([first, before, after]):
            pipe_index = len(self.pipeline)
            self.pipeline.append(pipe)
        elif first:
            self.pipeline.insert(0, pipe)
        elif before and before in self.pipe_names:
            pipe_index = self.pipe_names.index(before)
            self.pipeline.insert(pipe_index, pipe)
        elif after and after in self.pipe_names:
            pipe_index = self.pipe_names.index(after) + 1
            self.pipeline.insert(pipe_index, pipe)
        else:
            msg = "No component '{}' found in pipeline. Available names: {}"
            raise ValueError(
                msg.format(before or after, self.pipe_names)
            )

    def remove_pipe(self, name: str):
        """
        Remove component from the pipeline.
        
        Parameters
        ----------
        name : str
            Name fo the component to remove.
            Ignored if not found.
        """
        try:
            i = self.pipe_names.index(name)
            self.pipeline.pop(i)
        except ValueError:
            pass

    def has_pipe(self, name):
        """
        Is component in the pipeline?
        
        Parameters
        ----------
        name : str
            Name fo the component to find.

        Returns
        -------
        is_in : bool
        """
        return name in self.pipe_names

    def fit(self, X, y=None):
        """
        Fit the pipeline.

        Parameters
        ----------
        X : Iterable[str]
            Iterable of strings

        y : NoneType
            Only there for scikit-learn compatibility

        Returns
        -------
        self : TextPreprocessor
        """
        if hasattr(self, '_swc'):
            self._swc.fit(X)
        return self

    def transform(self, X, *_):
        """
        Apply the preprocessor on X.

        Parameters
        ----------
        X : Iterable[str]
            Iterable of strings

        Returns
        -------
        X : Iterable[str]
        """
        X = pd.Series(X)

        def flatten_series(list_of_series):
            "Flatten a list of pd.Series to a combined Series"
            return pd.concat(list_of_series, ignore_index=True, copy=False).values

        if self.n_jobs == 1:
            return X.apply(self._preprocess_text)
        else:
            return jl.preprocess_parallel(
                X, self._preprocess_part, n_jobs=self.n_jobs,
                flatten=flatten_series, chunksize=self.batch_size)

    def _preprocess_part(self, part):
        """
        Apply the :func:`TextPreprocessor._preprocess_text`
        function on a part of data. Used for parallelelizing
        the pipeline.

        Parameters
        ----------
        part : Iterable[str]
            Iterable of strings

        Returns
        -------
        part : Iterable[str]
        """
        return part.apply(self._preprocess_text)

    def _preprocess_text(self, text):
        """
        Apply the pipeline functions in sequence
        on the input string.

        Parameters
        ----------
        text : str
            Input string

        Returns
        -------
        text : str
        """
        for _, func in self.pipeline:
            text = func(text)
        return text
