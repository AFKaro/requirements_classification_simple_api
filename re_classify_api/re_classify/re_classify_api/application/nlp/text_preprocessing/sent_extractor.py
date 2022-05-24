from src.domain.model.user_redaction import Redaction
from typing import List
import nltk as nlp


class Extractor():

    def extract(redaction: str) -> List[str]:
        return nlp.sent_tokenize(redaction)
