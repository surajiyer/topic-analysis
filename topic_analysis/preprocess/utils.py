import importlib.resources as pkg_resources
import pandas as pd
from typing import Dict, List, Iterable


class SimpleDutchEnglishDetector:
    """
    Detect if language of input string is English (en) or Dutch (nl).
    Returns 'unk' if count of both language tokens is 0.

    Examples
    --------
    >>> lang_detect = SimpleDutchEnglishDetector()
    >>> lang_detect('this is a test'.split())
    'en'
    >>> lang_detect('dit is een test'.split())
    'nl'
    """
    SUPPORTED_LANGUAGES = ['nl', 'en']

    def __init__(
            self, vocab_files_path: Dict[str, List[str]] = dict(),
            term_index=0, separator=' ', encoding=None):
        """
        Parameters
        ----------
        vocab_files_path : Dict[key : str, List[str]]
            List of file paths to vocabulary to perform language detection.
            `key` must be in ('en', 'nl'). Vocabulary files must be one
            token per line.

        term_index : int, default=0
            Column index of the terms if there are multiple columns

        separator : str, default=' '
            Separator character if there are multiple columns

        encoding : str, default=None
            Text encoding of all text files listed in `vocab_files_path`.
        """

        assert len(set(vocab_files_path.keys()).difference(self.SUPPORTED_LANGUAGES)) == 0\
            , f'`vocab_files_path.keys()={vocab_files_path.keys()}` must be in {self.SUPPORTED_LANGUAGES}'

        # load Dutch vocabulary
        self.language_dictionary = dict()
        with pkg_resources.path('topic_analysis.preprocess.data', 'nl_50k.txt') as fp:
            self.language_dictionary['nl'] = set(pd.read_csv(fp, sep=' ').iloc[:, 0])

        # load custom Dutch vocabulary if given
        if vocab_files_path.get('nl', None):
            for fp in vocab_files_path['nl']:
                self.language_dictionary['nl'] |= set(
                    pd.read_csv(fp, sep=separator, encoding=encoding).iloc[:, term_index])

        # load English vocabulary
        with pkg_resources.path('topic_analysis.preprocess.data', 'en_50k.txt') as fp:
            self.language_dictionary['en'] = set(pd.read_csv(fp, sep=' ').iloc[:, 0])

        # load custom English vocabulary if given
        if vocab_files_path.get('en', None):
            for fp in vocab_files_path['en']:
                self.language_dictionary['en'] |= set(pd.read_csv(
                    fp, sep=separator, encoding=encoding).iloc[:, term_index])

    def __call__(self, tokens: Iterable[str]) -> str:
        """
        Count words against English versus Dutch dictionaries.

        Parameters
        ----------
        tokens : Iterable[str]

        Returns
        -------
        lang_iso_code : str
            two-letter language ISO code or 'unk' for unknown
        """
        count = {k: 0 for k in self.SUPPORTED_LANGUAGES}
        for w in tokens:
            w = w.lower()
            count['nl'] += int(w in self.language_dictionary['nl'])
            count['en'] += int(w in self.language_dictionary['en'])

        return 'unk'\
            if (count['nl'] == count['en'] == 0)\
            else max(count, key=count.get)


def get_component_name(component):
    if hasattr(component, "name"):
        return component.name
    if hasattr(component, "__name__"):
        return component.__name__
    if hasattr(component, "__class__") and hasattr(component.__class__, "__name__"):
        return component.__class__.__name__
    return repr(component)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
