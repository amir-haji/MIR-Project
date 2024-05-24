import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


class FastTextDataLoader:
    """
    This class is designed to load and pre-process data for training a FastText model.

    It takes the file path to a data source containing movie information (synopses, summaries, reviews, titles, genres) as input.
    The class provides methods to read the data into a pandas DataFrame, pre-process the text data, and create training data (features and labels)
    """
    def __init__(self, file_path, preprocess):
        """
        Initializes the FastTextDataLoader class with the file path to the data source.

        Parameters
        ----------
        file_path: str
            The path to the file containing movie information.
        """
        self.file_path = file_path
        self.preprocess = preprocess

    def read_data_to_df(self):
        """
        Reads data from the specified file path and creates a pandas DataFrame containing movie information.

        You can use an IndexReader class to access the data based on document IDs.
        It extracts synopses, summaries, reviews, titles, and genres for each movie.
        The extracted data is then stored in a pandas DataFrame with appropriate column names.

        Returns
        ----------
            pd.DataFrame: A pandas DataFrame containing movie information (synopses, summaries, reviews, titles, genres).
        """
        with open(self.file_path, 'r') as f:
            data = json.loads(f.read())
            f.close()
        data = data[:100]
        data_dict = {'synposis': [], 'summaries': [], 'reviews': [], 'title': [], 'genres': []}
        
        with tqdm(data) as t:
            for movie in t:
                if (movie['synposis'] is None and \
                movie['summaries'] is None and \
                movie['reviews'] is None and \
                movie['title'] is None) or \
                len(movie['genres']) == 0 :
                    continue

                data_dict['synposis'].append(' '.join(self.preprocess(x) for x in ([''] if movie['synposis'] is None else movie['synposis'])).strip())
                data_dict['summaries'].append(' '.join(self.preprocess(x) for x in ([''] if movie['summaries'] is None else movie['summaries'])).strip())
                data_dict['reviews'].append(' '.join(self.preprocess(x[0]) for x in ([['', '']] if movie['reviews'] is None else movie['reviews'])).strip())
                data_dict['title'].append(self.preprocess('' if movie['title'] is None else movie['title']))
                data_dict['genres'].append(self.preprocess(movie['genres'][0]))
                
        return pd.DataFrame(data_dict)
                
                
    def create_train_data(self):
        """
        Reads data using the read_data_to_df function, pre-processes the text data, and creates training data (features and labels).

        Returns:
            tuple: A tuple containing two NumPy arrays: X (preprocessed text data) and y (encoded genre labels).
        """
    
        df = self.read_data_to_df()
        
        self.le = LabelEncoder()
        df['genres'] = self.le.fit_transform(df['genres'])
        self.mapping = dict(zip(range(len(self.le.classes_)), self.le.classes_))

        df['text'] = df['synposis'] + ' ' + df['summaries']  + ' ' + df['reviews'] + ' ' + df['title']
        X = np.array(df['text'])
        y = np.array(df['genres'])
        
        return X, y
