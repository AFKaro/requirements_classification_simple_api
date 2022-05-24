from nltk.stem import WordNetLemmatizer
import nltk as nlp


class TextPreprocessing:

    def run(self, text: str = "") -> list:
        str_help = ""
            
        if text:
            str_help = text.lower()
            str_help = self.tokenization(str_help)
            str_help = self.normalization(str_help)
            str_help = self.lemmatization(str_help)
        return str_help


    def tokenization(self, text):
        return nlp.word_tokenize(text)


    def normalization(self, text):
        stop_words = nlp.corpus.stopwords.words('english')
        return [word for word in text if word not in stop_words and word.isalpha()]


    def lemmatization(self, text):
        lemma = WordNetLemmatizer()
        return [lemma.lemmatize(word) for word in text]