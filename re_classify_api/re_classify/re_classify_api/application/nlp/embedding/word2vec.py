from application.nlp.embedding.embedding_interface import EmbeddingInterface
from infra import local_data
import numpy as np


class W2V(EmbeddingInterface):
    """"""

    def vectorize(tokens: list):
        file_name = "GoogleNews-vectors-negative300.bin"
        model = local_data.get_word2vec_model(file_name)
        sentence_vector = []
        for token in tokens:
            try:
                sentence_vector.append(model[token])
            except:
                sentence_vector.append(np.zeros(300))
        return sentence_vector
