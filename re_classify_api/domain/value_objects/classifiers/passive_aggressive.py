from skopt.space import Real, Categorical, Integer
from sklearn.linear_model import PassiveAggressiveClassifier
from src.domain.value_objects.classifiers.classifier_interface import ClassifierInterface


class PassiveAggressive(ClassifierInterface):

    def __init_subclass__(cls) -> None:
        cls.name = "Passive Aggressive"
        cls.initials = "PA"
        cls.model = PassiveAggressiveClassifier()
        cls.search_spaces = {
            'tol': Real(1e-5, 1e-3),
            'C': Real(1e-2, 1e1),
            'fit_intercept': [True, False],
            'max_iter': Integer(1000, 1500),
            'early_stopping': [True],
            'validation_fraction': [0.1, 0.2],
            'n_iter_no_change': [5, 10],
            'loss': ['hinge', 'squared_hinge'],
            'warm_start': [True, False],
            'n_jobs': [-1]
        }
        return super().__init_subclass__()
