from dataclasses import dataclass
from sklearn.cluster import KMeans


@dataclass
class KMeansModel:

    model: KMeans

    def fit(self, train_data: list, n_clusters: int):
        self.model = KMeans(n_clusters=n_clusters).fit(train_data)

    def predict(self, test_data: list) -> list:
        return self.model.predict(test_data)
