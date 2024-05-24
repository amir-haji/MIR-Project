import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm
from scipy.spatial import distance
from basic_classifier import BasicClassifier
from data_loader import ReviewLoader


class KnnClassifier(BasicClassifier):
    def __init__(self, n_neighbors):
        super().__init__()
        self.k = n_neighbors

    def fit(self, x, y):
        """
        Fit the model using X as training data and y as target values
        use the Euclidean distance to find the k nearest neighbors
        Warning: Maybe you need to reduce the size of X to avoid memory errors

        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        self
            Returns self as a classifier
        """
        self.x_train = x
        self.y_train = y

    def predict(self, x):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        Returns
        -------
        np.ndarray
            Return the predicted class for each doc
            with the highest probability (argmax)
        """
        
        prediction = []
        with tqdm(x) as t:
            for z in t:
                distances = []
                for j in range(len(self.x_train)):
                    distances.append((j, distance.euclidean(z, x_train[j])))

                distances.sort(key = lambda x: x[1])
                neighbours = distances[:self.k]
                labels = [y_train[x[0]] for x in neighbours]

                if sum(labels) > len(labels)/2:
                    prediction.append(1)
                else:
                    prediction.append(0)
                
        return np.array(prediction)
            
            
        
    def prediction_report(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        str
            Return the classification report
        """
        y_pred = self.predict(x)
        print(classification_report(y, y_pred))



# F1 Accuracy : 70%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    rl = ReviewLoader('IMDB Dataset.csv')
    rl.load_data()
    rl.get_embeddings()

    x_train, x_test, y_train, y_test = rl.split_data()

    cls = KnnClassifier(5)
    cls.fit(x_train, y_train)
    cls.prediction_report(x_test, y_test)
