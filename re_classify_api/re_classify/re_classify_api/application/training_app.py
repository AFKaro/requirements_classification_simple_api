from typing import List
from nlp.embedding import vectorizer
from nlp.embedding.tfidf import Tfidf
from domain.value_objects.requirement import Requirement
from ml.classification_task.classification import Classification
from nlp.text_preprocessing.preprocessing import TextPreprocessing
from domain.value_objects.classifiers.passive_aggressive import PassiveAggressive


class TrainingApp:

    def run(self, requirements: List[Requirement]):
        requirements_text = []
        labels = []
        for requirement in requirements:
            text = requirement.__getattribute__("text")
            label = requirement.__getattribute__("label")
            p_requirement = TextPreprocessing().run(text)
            v_requirement = vectorizer.run(Tfidf, p_requirement)
            requirements_text.append(v_requirement)
            labels.append(label)
        Classification().training(
            x=requirements_text,
            y=labels,
            classifier=PassiveAggressive
        )
