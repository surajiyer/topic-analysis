# Topic analysis
This is a Python library to perform topic detection on textual data that are generated over time. The temporal nature of the documents are taken into account in the topic mining process using BNgrams.

## Installation
1. Assuming Git is configured correctly to work with GitLab, copy the clone URL.
2. Open Git Bash or git-enabled CMD locally and run the following:
```
cd path/to/projects/directory
git clone copied-clone-url
```
3. Open python-enabled CMD locally and run the following:
```
cd path/to/projects/directory/topic-analysis
pip install -e . --user
```

## Usage
For a quick tutorial on how to use the library, refer to the notebooks in [this](notebooks) folder. Detailed information on the API can be found [here](build/docs/content/api-documentation.md).

## How it works?

1. **Text preprocessing**: Preprocess the text data to clean it and make it easier to work with. Some of the steps we can do with the `topic_analysis.preprocess.text.TextPreprocessor` are: lowercase the text, regex clean, spell check, fix word compounding, remove stopwords, lemmatize or stem, parse text. Internally the `TextPreprocessor` builds a pipeline (list) of functions which take a string as input and give back a string as output which is executed on the input text data in sequence. Additional custom preprocessing steps can be be added to the pipeline, and existing ones can be modified or removed.

2. **Topic detection**: Detect topics, i.e, cluster of key phrases, and assign topics IDs to input documents.

    1. **Phrase extraction**: Candidate phrases can be either be simple n-grams or noun phrases extracted by matching over POS tags.

    2. **Document vectorization**: Documents are vectorized with $`DF-IDF_{t}`$ (BNgram).

    3. **Topic clustering**: Phrases are clustered into topics based on co-occurence with other phrases across all documents. There are two method to clustering phrases [2]:

        - *Hierarchical Agglomerative Clustering (HAC)* using phrase co-occurence confidence as similarity function for pairs of phrases. HAC starts with as many cluster as the number of unique extracted phrases. Then starts grouping them into clusters until a distance threshold is reached which needs to be set manually at the beginning. By default, clusters stop getting merged when the furthest pairs of phrases in each cluster do not co-occur in at least 50% of the documents in which either phrase occurs. The output clusters of terms are flattened and represent unique topics.

        - *Gaussian Mixture modeling (GMM)* is a more generalized version of the K-means clustering algorithm which learns a Gaussian posterior distribution per cluster / topic from the term-document co-occurence matrix. This algorithm does require the number of clusters to be specified at the start however this is pseudo-automated by setting a range of values for the cluster sizes and each size will be evaluated using the Bayesian Information Criterion (BIC) score for quality of model fit to the data. If the number of clusters is over-specified, this is still fine because the model will fit mostly outlier terms to the extra unnecessary topics which will be ranked lower in the next step.

    4. **Topic ranking**: Topics can be ranked based on the maximum BNgram score over all phrases present within a topic.

    5. **Document-topic assignment**: Topics are assigned to documents based on usage of topic phrases in the documents.

3. **Trends view**: Plot individual phrases or entire topics usage over time to explain trends.

## References
BNgrams document vectorization
```
[1] @inproceedings{martin2014real,
    title={Real-time topic detection with bursty n-grams: RGU's submission to the 2014 SNOW Challenge.},
    author={Martin, Carlos and Goker, Ayse},
    year={2014},
    organization={CEUR Workshop Proceedings}
}
```

Topic detection using BNgrams
```
[2] @incollection{martin2015mining,
  title={Mining newsworthy topics from social media},
  author={Martin, Carlos and Corney, David and Goker, Ayse},
  booktitle={Advances in social media analysis},
  pages={21--43},
  year={2015},
  publisher={Springer}
}
[3] @article{winarko2019trending,
  title={Trending topics detection of Indonesian tweets using BN-grams and Doc-p},
  author={Winarko, Edi and Pulungan, Reza and others},
  journal={Journal of King Saud University-Computer and Information Sciences},
  volume={31},
  number={2},
  pages={266--274},
  year={2019},
  publisher={Elsevier}
}
```

## Roadmap
This is a research tool and will continue to be actively developed.