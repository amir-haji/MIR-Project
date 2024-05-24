import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append('/Users/hajmohammadrezaee/Desktop/MIR-Project/Logic/core')
from word_embedding.fasttext_model import FastText, preprocess_text


class ReviewLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.fasttext_model = None
        self.review_tokens = []
        self.preprocessed_reviews = []
        self.sentiments = []
        self.embeddings = []

    def load_data(self):
        """
        Load the data from the csv file and preprocess the text. Then save the normalized tokens and the sentiment labels.
        Also, load the fasttext model.
        """
        df = pd.read_csv(self.file_path)
        
        with tqdm(df['review']) as t:
            for x in t:
                normalized_text = preprocess_text(x)
                self.preprocessed_reviews.append(normalized_text)
                for token in normalized_text.split(' '):
                    if token not in self.review_tokens:
                        self.review_tokens.append(token)
                        
        self.le = LabelEncoder()
        df['sentiment'] = self.le.fit_transform(df['sentiment'])
        self.sentiments = list(df['sentiment'])
        
        self.fasttext_model = FastText()
        self.fasttext_model.prepare(None, 'load', path = 'FastText_model.bin')
            

    def get_embeddings(self):
        """
        Get the embeddings for the reviews using the fasttext model.
        """
        with tqdm(self.preprocessed_reviews) as t:
            for text in t:
                self.embeddings.append(self.fasttext_model.get_query_embedding(text))

    def split_data(self, test_data_ratio=0.2):
        """
        Split the data into training and testing data.

        Parameters
        ----------
        test_data_ratio: float
            The ratio of the test data
        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
            Return the training and testing data for the embeddings and the sentiments.
            in the order of x_train, x_test, y_train, y_test
        """
        return train_test_split(np.array(self.embeddings), np.array(self.sentiments), test_size = test_data_ratio, random_state = 42)


