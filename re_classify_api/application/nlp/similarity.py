from text_preprocessing.preprocessing import TextPreprocessing
from sklearn.metrics.pairwise import cosine_similarity
from src.domain.nlp.embedding import vectorizer
from src.domain.nlp.embedding import word2vec
from dataclasses import dataclass


@dataclass
class Similarity:

    text_1: str = None
    text_2: str = None

    def calculate_similarity(self) -> float:

        tokes_txt1 = TextPreprocessing().run(self.text_1)
        tokes_txt2 = TextPreprocessing().run(self.text_2)

        vector_txt1 = vectorizer.run(word2vec, tokes_txt1)
        vector_txt2 = vectorizer.run(word2vec, tokes_txt2)

        return cosine_similarity(vector_txt1, vector_txt2).mean()
