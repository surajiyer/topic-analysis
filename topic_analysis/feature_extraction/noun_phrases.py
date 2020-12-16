"""
Extract noun phrases from text.
Based on https://github.com/slanglab/phrasemachine.

References
----------
1. Bag of What? Simple Noun Phrase Extraction for Text Analysis
    a. paper: https://www.aclweb.org/anthology/W16-5615.pdf
    b. slides: http://brenocon.com/oconnor_textasdata2016.pdf
2. Fightin’ Words: Lexical Feature Selection and Evaluation for Identifying the Content of Political Conflict
    a. paper: http://languagelog.ldc.upenn.edu/myl/Monroe.pdf
"""
import re
from spacy import displacy
# from spacy.matcher import Matcher
from spacy.tokens import Doc, Span
from spacy.util import filter_spans as fs


coarsemap = {
    'A': "JJ JJR JJS CoarseADJ CD CoarseNUM A ADV ADJ".split(),
    'D': "DT CoarseDET D DET".split(),
    'P': "IN TO CoarseADP P ADP".split(),
    'N': "NN NNS NNP NNPS FW CoarseNOUN N NOUN PROPN ^".split(),
    'V': "VERB".split()
    # all other tags get O
}
tag2coarse = {}
for coarsetag, inputtags in coarsemap.items():
    for intag in inputtags:
        assert intag not in tag2coarse
        tag2coarse[intag] = coarsetag


def coarse_tag_str(pos_seq):
    """Convert POS sequence to our coarse system, formatted as a string."""
    global tag2coarse
    tags = [tag2coarse.get(tag, 'O') for tag in pos_seq]
    return ''.join(tags)


def extract_ngram_filter(pos_seq, regex, minlen=1, maxlen=8):
    """The "FilterFSA" method in Handler et al. 2016.
    Returns token position spans of valid ngrams."""
    ss = coarse_tag_str(pos_seq)
    def gen():
        for s in range(len(ss)):
            for n in range(minlen, 1 + min(maxlen, len(ss) - s)):
                e = s+n
                substr = ss[s:e]
                if re.match(regex + "$", substr):
                    yield (s, e)
    return list(gen())


class NounPhraseMatcher:
    """
    Spacy module for extracting noun phrases from text.
    """

    name = 'noun_phrase_matcher'
#     noun_chunk_pattern = [
#         {'POS': 'ADJ', 'OP': '*'},
#         {'POS': {'REGEX': '(PROPN|NOUN)'}, 'OP': '+'}]

    # The grammar!
    # SimpleNP (Spacy version): (ADJ|NOUN|PROPN)*(NOUN|PROPN)(ADPDET*(ADJ|NOUN|PROPN)*(NOUN|PROPN))*
    SimpleNP = "(A|N)*N(PD*(A|N)*N)*"
    VP = "A*V"

    def __init__(
#             self, nlp, lowercase=False, lemmatize=False, filter_spans=True,
            self, lowercase=False, lemmatize=False, filter_spans=False,
            include_verb_phrases=False, minlen=1, maxlen=8, attr_name=None,
            force_extension=True):
        """
        Parameters
        ----------
        lowercase : bool, default=False
            Lowercase output phrases. Not applied to input
            before extracting phrases. Duplicates (if any)
            after lowercasing  will be removed.

        lemmatize : bool, default=False
            Lemmatize output phrases. Not applied to input
            before extracting phrases. Duplicates (if any)
            after lemmatizing will be removed.

        filter_spans : bool, default=False
            Removes duplicates or overlappping phrases from
            the output.

        include_verb_phrases : bool, default=False
            Indicator to include verb phrases also.

        minlen : int, default=1
            Minimum length of extracted multi-word phrases.
            Used for tokenizing the text.

        maxlen : int, default=8
            Maximum length of extracted multi-word phrases.
            Used for tokenizing the text.

        attr_name : str, default=None
            Attribute name to set on the ._ property. If None,
            by default, phrases will get stored to `doc._.noun_phrases`.

        force_extension : bool, default=True
            A boolean value to force recreate the `_.[attr_name]` attribute
            if already exists.
        """
        # build matcher for getting noun phrases
#         self.matcher = Matcher(nlp.vocab)
#         self.matcher.add("noun_chunk", None, self.noun_chunk_pattern)
        self.attr_name = 'noun_phrases' if attr_name is None else attr_name
        Doc.set_extension(self.attr_name, default=[], force=force_extension)
        self.lowercase = lowercase
        self.lemmatize = lemmatize
        self.filter_spans = filter_spans
        self.include_verb_phrases = include_verb_phrases
        self.minlen = minlen
        self.maxlen = maxlen

    def __call__(self, doc: Doc):

        def process(text: Span):
            if self.lemmatize:
                text = text.lemma_
            text = str(text)
            if self.lowercase:
                text = text.lower()
            return text

        postags = [token.pos_ for token in doc]

        if self.filter_spans:
            phrases = [
                process(p) for p in fs(
                    [doc[start:end] for start, end in extract_ngram_filter(
                        postags, self.SimpleNP, self.minlen, self.maxlen)])]
            if self.include_verb_phrases:
                phrases += [
                    process(p) for p in fs(
                        [doc[start:end] for start, end in extract_ngram_filter(
                            postags, self.VP, self.minlen, self.maxlen)])]
            phrases = list(set(phrases))
        else:
            phrases = [
                process(doc[start:end]) for start, end in extract_ngram_filter(
                    postags, self.SimpleNP, self.minlen, self.maxlen)]
            if self.include_verb_phrases:
                phrases += [
                    process(doc[start:end]) for start, end in extract_ngram_filter(
                        postags, self.VP, self.minlen, self.maxlen)]
            phrases = list(set(phrases))

        # remove smaller of the overlapping candidates
        # TODO: change this naive implementation of word overlap checking
#         remaining_phrases = phrases[:]
#         for p1 in phrases:
#             for p2 in remaining_phrases[:]:
#                 if p1 == p2:
#                     continue
#                 elif p1 in remaining_phrases and p1 in p2:
#                     remaining_phrases.remove(p1)
#                 elif p2 in p1:
#                     remaining_phrases.remove(p2)
        doc._.set(self.attr_name, phrases)

        return doc

    def render_with_displacy(self, doc: Doc, **displacy_render_kws):
        """
        Serve visualization of sentences containing match with displaCy
        Set `jupyter=True` if running within jupyter environment.
        """
        postags = [token.pos_ for token in doc]
        ents = [(
            lambda span: {
                "start": span.start_char,
                "end": span.end_char,
                "label": 'NP'})(doc[start:end])
            for start, end in extract_ngram_filter(
                postags, self.SimpleNP, self.minlen, self.maxlen)]
        if self.include_verb_phrases:
            ents += [(
                lambda span: {
                    "start": span.start_char,
                    "end": span.end_char,
                    "label": 'VP'})(doc[start:end])
                for start, end in extract_ngram_filter(
                    postags, self.VP, self.minlen, self.maxlen)]

        # set manual=True to make displaCy render straight from a dictionary
        displacy_render_kws.pop('manual', None)
        displacy_render_kws.pop('style', None)
        return displacy.render(
            {"text": doc.text, "ents": ents},
            style="ent", manual=True, **displacy_render_kws)
