import logging
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

from re_classify_api.domain.value_objects.classifiers.classifier_interface import ClassifierInterface


def get_optimized_model(classifier: ClassifierInterface,  x, y):

    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
    search_spaces = classifier.get_search_spaces()
    estimator = classifier.get_instance()
    model_name = classifier.get_name()

    logging.info("\nOptimizing " + model_name + " algorithm...")

    model = BayesSearchCV(estimator=estimator, search_spaces=search_spaces,
                          n_iter=150, scoring='f1_macro', cv=cv, refit=True, return_train_score=True,
                          n_jobs=-1, n_points=3, pre_dispatch=3)
    model.fit(x, y)
    return model.best_estimator_
