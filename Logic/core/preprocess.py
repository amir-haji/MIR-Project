from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import contractions
import unidecode
import string
import re


class Preprocessor:

    def __init__(self, documents: list):
        """
        Initialize the class.

        Parameters
        ----------
        documents : list
            The list of documents to be preprocessed, path to stop words, or other parameters.
        """
        # TODO
        with open('/Users/hajmohammadrezaee/Desktop/MIR-Project/UI/../Logic/core/stopwords.txt', 'r') as f:
            txt = f.read()
            f.close()
            
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.documents = documents
        self.stopwords = txt.split('\n')

    def preprocess(self):
        """
        Preprocess the text using the methods in the class.

        Returns
        ----------
        List[str]
            The preprocessed documents.
        """
         # TODO
        for doc in self.documents:
            if isinstance(doc, str):
                self.documents = [self.normalize(doc)]
                return self.documents.copy()
            else:
                for k in doc:
                    if isinstance(doc[k], list):
                        new_item = []
                        if len(doc[k]) != 0:
                            if isinstance(doc[k][0], list):
                                # reviews
                                for review, score in doc[k]:
                                    new_item.append([self.normalize(review), score])
                            else:
                                new_item = [self.normalize(text) for text in doc[k]]
                        doc[k] = new_item
                    else:
                        doc[k] = self.normalize(doc[k])
                    
        return

    def normalize(self, text: str):
        """
        Normalize the text by converting it to a lower case, stemming, lemmatization, etc.

        Parameters
        ----------
        text : str
            The text to be normalized.

        Returns
        ----------
        str
            The normalized text.
        """
        # TODO

        text = unidecode.unidecode(text)
        text = text.lower()
        
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text.strip()

        sentences = sent_tokenize(text)
        normalized_text = ''
        for sentence in sentences:
            sent = self.remove_links(sentence)
            sent = self.remove_punctuations(sent)
            words = self.remove_stopwords(sent)
            normalized_text = normalized_text + ' ' + ' '.join(words)
        
        return normalized_text.strip()

    def remove_links(self, text: str):
        """
        Remove links from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with links removed.
        """
        patterns = [r'\S*http\S*', r'\S*www\S*', r'\S+\.ir\S*', r'\S+\.com\S*', r'\S+\.org\S*', r'\S*@\S*']
        # TODO

        for pattern in patterns:
            text = re.sub(pattern, '', text)
        return text

    def remove_punctuations(self, text: str):
        """
        Remove punctuations from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with punctuations removed.
        """
        # TODO
        punctuation_table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        text = text.translate(punctuation_table)
        return text

    def tokenize(self, text: str):
        """
        Tokenize the words in the text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        ----------
        list
            The list of words.
        """
        # TODO
        return word_tokenize(text)

    def remove_stopwords(self, text: str):
        """
        Remove stopwords from the text.

        Parameters
        ----------
        text : str
            The text to remove stopwords from.

        Returns
        ----------
        list
            The list of words with stopwords removed.
        """
        # TODO
        words = self.tokenize(text)
        final_words = []
        for w in words:
            if w not in self.stopwords:
                final_words.append(contractions.fix(self.lemmatizer.lemmatize(w)))
        return final_words

