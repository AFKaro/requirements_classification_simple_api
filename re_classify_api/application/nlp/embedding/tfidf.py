from re_classify_api.application.nlp.embedding.embedding_interface import EmbeddingInterface
from sklearn.feature_extraction.text import TfidfVectorizer
from dataclasses import dataclass
from typing import List


@dataclass
class Tfidf(EmbeddingInterface):
    """"""

    vectorizer: TfidfVectorizer

    def fit(self, tokens: List):
        self.vectorizer.fit(tokens)
        return self.vectorizer

    def vectorize(self, tokens: list):
        return self.vectorizer.transform(tokens)

    def key_words(self):
        return self.vectorizer.get_feature_names_out()
