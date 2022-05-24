from re_classify_api.application.nlp.text_preprocessing.preprocessing import TextPreprocessing
from re_classify_api.infra.local_data import get_dataset, export_sklearn_model
from sklearn.feature_extraction.text import TfidfVectorizer
from re_classify_api.application.nlp.embedding.tfidf import Tfidf


class EmbeddingApp:

    def fit_tfidf():
        corpus = get_dataset()
        x = list()
        for text in corpus["RequirementText"]:
            x.append(
                " ".join(TextPreprocessing().run(text))
            )
        tfidf_model = Tfidf(vectorizer=TfidfVectorizer()).fit(x)
        export_sklearn_model(tfidf_model, "models/tfidf_model.joblib")
