import numpy as np
from typing import List
import wandb
import json
import os

class Evaluation:

    def __init__(self, name: str):
            self.name = name

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The precision of the predicted results
        """
        precision = 0.0

        # TODO: Calculate precision here
        
        for i in range(len(predicted)):
            TP = len(set(predicted[i]).intersection(set(actual[i])))
            precision += TP / len(predicted[i])
            
        precision /= len(predicted)
        
        return precision
    
    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the recall of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The recall of the predicted results
        """
        recall = 0.0

        # TODO: Calculate recall here
        
        for i in range(len(predicted)):
            TP = len(set(predicted[i]).intersection(set(actual[i])))
            recall += TP / len(actual[i])
            
        recall /= len(predicted)
        return recall
    
    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the F1 score of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The F1 score of the predicted results    
        """
        f1 = 0.0

        # TODO: Calculate F1 here
        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        
        f1 = (2 * precision * recall) / (precision + recall)

        return f1
    
    def calculate_AP(self, actual: List[str], predicted: List[str]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Average Precision of the predicted results
        """
        AP = 0.0

        # TODO: Calculate AP here
        
        num_corrects = 0
        for i in range(len(predicted)):
            if predicted[i] in actual:
                num_corrects += 1
                AP += num_corrects / (i + 1)

        
        AP /= num_corrects
        return AP
    
    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Mean Average Precision of the predicted results
        """
        MAP = 0.0

        # TODO: Calculate MAP here
        
        for i in range(len(predicted)):
            MAP += self.calculate_AP(actual[i], predicted[i])
            
        MAP /= len(predicted)
        return MAP
    
    def calculate_DCG(self, actual: List[str], predicted: List[str]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The DCG of the predicted results
        """
        DCG = 0.0

        # TODO: Calculate DCG here
        
        relevance_score = {}
        for i in range(len(actual)):
            relevance_score[actual[i]] = len(actual) - i
            
        for i in range(len(predicted)):
            score = relevance_score.get(predicted[i], 0)
            DCG += score if i == 0 else score/np.log2(i + 1)

        return DCG
    
    def calculate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The NDCG of the predicted results
        """
        NDCG = 0.0

        # TODO: Calculate NDCG here
        
        for i in range(len(predicted)):
            ideal_score = self.calculate_DCG(actual[i], actual[i])
            DCG_score = self.calculate_DCG(actual[i], predicted[i])
            NDCG += DCG_score / ideal_score
            
        NDCG /= len(predicted)
        return NDCG
    
    def calculate_RR(self, actual: List[str], predicted: List[str]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Reciprocal Rank of the predicted results
        """
        RR = 0.0

        # TODO: Calculate MRR here
        
        for i in range(len(predicted)):
            if predicted[i] in actual:
                RR = 1/(i + 1)
                break

        return RR
    
    def calculate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The MRR of the predicted results
        """
        MRR = 0.0

        # TODO: Calculate MRR here
        
        for i in range(len(predicted)):
            MRR += self.calculate_RR(actual[i], predicted[i])
            
        MRR /= len(predicted)
        return MRR
    

    def print_evaluation(self, precision, recall, f1, map, ndcg, mrr):
        """
        Prints the evaluation metrics

        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        print(f"name = {self.name}")

        #TODO: Print the evaluation metrics
        
        print(f'precision score = {precision}')
        print(f'recall score = {recall}')
        print(f'F1 score = {f1}')
        print(f'Mean Average Precision score = {map}')
        print(f'Normalized Discounted Cumulative Gain score = {ndcg}')
        print(f'Mean Reciprocal Rank score = {mrr}')
      

    def log_evaluation(self, precision, recall, f1, map, ndcg, mrr):
        """
        Use Wandb to log the evaluation metrics
      
        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        
        #TODO: Log the evaluation metrics using Wandb
        
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project= f'results {self.name}')
        logs = {'precision': precision,
               'recall': recall,
               'f1': f1,
               'map': map,
               'ndcg': ndcg,
               'mrr': mrr
               }
        
        wandb.log(logs)



    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        call all functions to calculate evaluation metrics

        parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results
            
        """

        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        map_score = self.calculate_MAP(actual, predicted)
        ndcg = self.calculate_NDCG(actual, predicted)
        mrr = self.calculate_MRR(actual, predicted)

        #call print and viualize functions
        self.print_evaluation(precision, recall, f1, map_score, ndcg, mrr)
        self.log_evaluation(precision, recall, f1, map_score, ndcg, mrr)

if __name__ == "__main__":
    with open('/Users/hajmohammadrezaee/Desktop/MIR-Project/Logic/core/utility/validation.json', 'r') as f:
        validation = json.loads(f.read())
        f.close()

    queries = list(validation.keys())
    actual = list(validation.values())

    with open('/Users/hajmohammadrezaee/Desktop/MIR-Project/prediction.json', 'r') as f:
        prediction = json.loads(f.read())
        f.close()

    predicted = list(prediction.values())
    eval = Evaluation('test')
    eval.calculate_evaluation(actual, predicted)



