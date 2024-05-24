import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_loader import ReviewLoader
from basic_classifier import BasicClassifier


class ReviewDataSet(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)

        if len(self.embeddings) != len(self.labels):
            raise Exception("Embddings and Labels must have the same length")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.embeddings[i], self.labels[i]


class MLPModel(nn.Module):
    def __init__(self, in_features=100, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, xb):
        return self.network(xb)


class DeepModelClassifier(BasicClassifier):
    def __init__(self, in_features, num_classes, batch_size, num_epochs=50):
        """
        Initialize the model with the given in_features and num_classes
        Parameters
        ----------
        in_features: int
            The number of input features
        num_classes: int
            The number of classes
        batch_size: int
            The batch size of dataloader
        """
        super().__init__()
        self.test_loader = None
        self.in_features = in_features
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = MLPModel(in_features=in_features, num_classes=num_classes)
        self.best_model = self.model.state_dict()
        self.best_test_accuracy = 0
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # self.device = 'mps' if torch.backends.mps.is_available else 'cpu'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def fit(self, x, y):
        """
        Fit the model on the given train_loader and test_loader for num_epochs epochs.
        You have to call set_test_dataloader before calling the fit function.
        Parameters
        ----------
        x: np.ndarray
            The training embeddings
        y: np.ndarray
            The training labels
        Returns
        -------
        self
        """
        
        train_dataset = ReviewDataSet(x, y)
        self.train_loader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True)
        
        for epoch in range(self.num_epochs):
            self.model.train()
            
            correct = 0
            total = 0
            total_loss = 0
            with tqdm(enumerate(self.train_loader), total = len(self.train_loader)) as pbar:
                for i, (embed, label) in pbar:
                    embed = embed.to(self.device)
                    label = label.to(self.device)
                    
                    self.optimizer.zero_grad()
                    
                    output = self.model(embed)
                    loss = self.criterion(output, label)
                    
                    total_loss += loss.item()
                    loss.backward()
                    self.optimizer.step()
                    
                    pred = nn.functional.softmax(output, dim=0).argmax(dim=1)
                    
                    total += len(label)
                    correct += torch.sum(pred == label).item()
                    
            print(f'[Train]: epoch={epoch}, avg_acc={100*(correct/total):.2f}, avg_loss={100*(total_loss/total):.2f}')
            
            f1, pred_label, true_label, eval_loss = self._eval_epoch(self.test_loader, self.model)
            test_accuracy = (np.sum(np.array(true_label) == np.array(pred_label)).item()) / len(true_label)
            
            if test_accuracy > self.best_test_accuracy:
                self.best_test_accuracy = test_accuracy
                self.best_model = self.model.state_dict()
                
            print(f'[Test]: avg_acc={100*(test_accuracy):.2f}, avg_loss={100*(eval_loss/total):.2f}')
                
            
        return self

    def predict(self, x):
        """
        Predict the labels on the given test_loader
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        Returns
        -------
        predicted_labels: list
            The predicted labels
        """
        x = torch.tensor(x).to(self.device)
        output = self.model(x)
        pred = nn.functional.softmax(output, dim=0).argmax(dim=1).cpu().numpy()
        
        return pred

    def _eval_epoch(self, dataloader: torch.utils.data.DataLoader, model):
        """
        Evaluate the model on the given dataloader. used for validation and test
        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
        Returns
        -------
        eval_loss: float
            The loss on the given dataloader
        predicted_labels: list
            The predicted labels
        true_labels: list
            The true labels
        f1_score_macro: float
            The f1 score on the given dataloader
        """
        
        print('Eval Model ...')
        eval_loss = 0
        total = 0
        true_labels = []
        pred_labels = []
        
        with tqdm(enumerate(self.test_loader), total = len(self.test_loader)) as pbar:
            for i, (embed, label) in pbar:
                embed = embed.to(self.device)
                label = label.to(self.device)
                
                output = self.model(embed)
                loss = self.criterion(output, label)
                
                total += len(label)
                eval_loss += loss.item()
                pred_labels.append(nn.functional.softmax(output, dim=0).argmax(dim=1))
                true_labels.append(label)
                
        pred_labels = list(torch.cat(pred_labels).cpu())
        true_labels = list(torch.cat(true_labels).cpu())
        eval_loss /= total
        
        f1 = f1_score(true_labels, pred_labels, average = 'macro')
        
        return eval_loss, pred_labels, true_labels, f1
    
    def set_test_dataloader(self, X_test, y_test):
        """
        Set the test dataloader. This is used to evaluate the model on the test set while training
        Parameters
        ----------
        X_test: np.ndarray
            The test embeddings
        y_test: np.ndarray
            The test labels
        Returns
        -------
        self
            Returns self
        """
        test_dataset = ReviewDataSet(X_test, y_test)
        self.test_loader = DataLoader(test_dataset, batch_size = self.batch_size, shuffle = False)
        return self

    def prediction_report(self, x, y):
        """
        Get the classification report on the given test set
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        y: np.ndarray
            The test labels
        Returns
        -------
        str
            The classification report
        """
        self.model.load_state_dict(self.best_model)
        y_pred = self.predict(x)
        
        print(classification_report(y, y_pred))

# F1 Accuracy : 79%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    rl = ReviewLoader('IMDB Dataset.csv')
    rl.load_data()
    rl.get_embeddings()

    x_train, x_test, y_train, y_test = rl.split_data()

    model = DeepModelClassifier(100, 2, 100)
    model.set_test_dataloader(x_test, y_test)
    model.fit(x_train, y_train)
    model.prediction_report(x_test, y_test)

