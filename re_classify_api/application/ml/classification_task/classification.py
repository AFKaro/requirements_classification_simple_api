import logging
import sklearn
from typing import List
from dataclasses import dataclass
from re_classify_api.infra.local_data import export_sklearn_model
from re_classify_api.application.ml.classification_task import optimization
from re_classify_api.domain.value_objects.classifiers.classifier_interface import ClassifierInterface


@dataclass
class Classification:

    inter: int
    k_folds: int
    model: sklearn
    classifier: ClassifierInterface

    def classify(self, x: str) -> str:
        return self.model.predict(x)

    def training(self, x, y):
        logging.info("\nTraining the algorithm...")
        optimized_model = optimization.get_optimized_model(
            self.classifier, x, y)
        export_sklearn_model(model=optimized_model,
                             file_name="models/lr_model.joblib")
