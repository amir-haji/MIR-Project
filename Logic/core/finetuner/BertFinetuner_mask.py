import transformers
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, Trainer, TrainingArguments
import evaluate
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BERTFinetuner:
    """
    A class for fine-tuning the BERT model on a movie genre classification task.
    """

    def __init__(self, file_path, top_n_genres=5):
        """
        Initialize the BERTFinetuner class.

        Args:
            file_path (str): The path to the JSON file containing the dataset.
            top_n_genres (int): The number of top genres to consider.
        """
        self.file_path = file_path
        self.top_n_genres = top_n_genres

        self.accuracy = evaluate.load('accuracy')
        self.precision = evaluate.load('precision')
        self.recall = evaluate.load('recall')
        self.f1 = evaluate.load('f1')

    def load_dataset(self):
        """
        Load the dataset from the JSON file.
        """
        with open('IMDB_crawled.json', 'r') as f:
          self.data = json.loads(f.read())
          f.close()


    def preprocess_genre_distribution(self):
        """
        Preprocess the dataset by filtering for the top n genres
        """
        genres = {}
        num_movie_genres = 0

        for movie in self.data:
          if len(movie['genres']) > 0:
            num_movie_genres += 1
            g = movie['genres'][0]
            if g in genres:
              genres[g] += 1
            else:
              genres[g] = 1

        sorted_results = sorted(genres.items(), key = lambda x: (x[1], x[0]), reverse = True)
        genres = [x[0] for x in sorted_results]
        freqs = [x[1] for x in sorted_results]

        plt.bar(genres, freqs, color = 'red')
        plt.xlabel('genres')
        plt.ylabel('frequency')
        plt.title('genres distributions visualization')
        plt.xticks(range(len(genres)), genres, rotation='vertical')
        plt.show()

    def split_dataset(self, sentences, labels, test_size=0.2, val_size=0.5):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the dataset to include in the validation split.
        """
        train_txt, test_txt, train_label, test_label = train_test_split(sentences, labels, test_size = test_size, random_state = 42)
        val_txt, test_txt, val_label, test_label = train_test_split(test_txt, test_label, test_size = val_size, random_state = 42)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        train_encodings = self.tokenizer(train_txt, truncation=True, padding=True)
        val_encodings = self.tokenizer(val_txt, truncation=True, padding=True)
        test_encodings = self.tokenizer(test_txt, truncation=True, padding=True)

        self.train_dataset = IMDbDataset(train_encodings, train_label)
        self.val_dataset = IMDbDataset(val_encodings, val_label)
        self.test_dataset = IMDbDataset(test_encodings, test_label)

        return self.train_dataset, self.val_dataset, self.test_dataset



    def create_dataset(self):
        """
        Create a PyTorch dataset from the given encodings and labels.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.

        Returns:
            IMDbDataset: A PyTorch dataset object.
        """
        sentences = []
        labels = []

        for movie in self.data:
          if len(movie['genres']) > 0 and movie['first_page_summary'] is not None:
            g = movie['genres'][0]
            summary = movie['first_page_summary']

            if g == 'Drama' or g == 'Romance':
              labels.append(0)
              sentences.append(summary)
            elif g == 'Action' or g == 'Thriller':
              labels.append(1)
              sentences.append(summary)
            elif g == 'Comedy':
              labels.append(2)
              sentences.append(summary)
            elif g == 'Animation':
              labels.append(3)
              sentences.append(summary)
            elif g == 'Crime':
              labels.append(4)
              sentences.append(summary)
            else:
              pass

        return self.split_dataset(sentences, labels)


    def fine_tune_bert(self, epochs=5, batch_size=16, warmup_steps=500, weight_decay=0.01):
        """
        Fine-tune the BERT model on the training data.

        Args:
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            weight_decay (float): The strength of weight decay regularization.
        """
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=64,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            )
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels = 5,

            output_attentions = False,
            output_hidden_states = False,
            )

        self.model.cuda()

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
        )

        self.trainer.train()


    def compute_metrics(self, pred):
        """
        Compute evaluation metrics based on the predictions.

        Args:
            pred (EvalPrediction): The model's predictions.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=1)
        result = {'accuracy': self.accuracy.compute(predictions=predictions, references=labels)['accuracy'],
              'precision': self.precision.compute(predictions=predictions, references=labels, average = 'macro')['precision'],
              'recall': self.recall.compute(predictions=predictions, references=labels, average = 'macro')['recall'],
              'f1': self.f1.compute(predictions=predictions, references=labels, average = 'macro')['f1']}
        return result

    def evaluate_model(self):
        """
        Evaluate the fine-tuned model on the test set.
        """
        print(self.trainer.evaluate(self.test_dataset))

    def save_model(self, model_name):
        """
        Save the fine-tuned model and tokenizer to the Hugging Face Hub.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub.
        """
        repo_name = f'hajimr80/{model_name}'
        self.tokenizer.push_to_hub(repo_name)
        self.model.push_to_hub(repo_name)

class IMDbDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for the movie genre classification task.
    """

    def __init__(self, encodings, labels):
        """
        Initialize the IMDbDataset class.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.
        """
        # TODO: Implement initialization logic

        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input encodings and labels.
        """
        # TODO: Implement item retrieval logic
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        # TODO: Implement length computation logic
        return len(self.labels)