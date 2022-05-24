from dataclasses import dataclass
import logging
from re_classify_api.infra.local_data import (get_dataset)
from re_classify_api.application.nlp.embedding import vectorizer
from re_classify_api.application.ml.classification_task.classification import Classification
from re_classify_api.application.nlp.embedding.embedding_interface import EmbeddingInterface
from re_classify_api.application.nlp.text_preprocessing.preprocessing import TextPreprocessing
from re_classify_api.models import Requirement
from re_classify_api.serializers import RequirementSerializer


@dataclass
class ClassificationApp:

    embedding: EmbeddingInterface
    classification: Classification

    def run(self, requirement: dict) -> str:
        text = requirement['text']
        txt_preprocessing = " ".join(TextPreprocessing().run(text))
        logging.info(f"Text preprocessing: {txt_preprocessing}")
        txt_vectorized = vectorizer.run(self.embedding, [txt_preprocessing])
        logging.info(f"Text vectorized: {txt_vectorized}")
        label = self.classification.classify(txt_vectorized)
       
        return label[0]

    def fit_model(self):
        dataset = get_dataset()
        data = dataset["RequirementText"]
        x = []
        y = dataset["Class"]
        for text in data:
            x.append(
                " ".join(TextPreprocessing().run(text))
            )
        x = vectorizer.run(self.embedding, x)
        self.classification.training(x, y)
