from re_classify_api.application.nlp.embedding.embedding_interface import EmbeddingInterface


def run(approach: EmbeddingInterface, tokens: list):
    """approach -> TF-IDF or Word2Vec"""
    return approach.vectorize(tokens)
