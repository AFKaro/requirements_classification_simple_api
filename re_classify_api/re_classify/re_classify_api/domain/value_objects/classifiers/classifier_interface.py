from dataclasses import dataclass
from skopt.space import Real, Categorical, Integer


@dataclass
class ClassifierInterface:

    name: str = None
    initials: str = None
    model: object = None
    instance: object = None
    search_spaces: dict = None
