import json
import numpy as np
from .utility.preprocess import Preprocessor
from .utility.scorer import Scorer
from .indexer import Indexes, Index_types, Index_reader


class SearchEngine:
    def __init__(self):
        """
        Initializes the search engine.

        """
        path = "/Users/hajmohammadrezaee/Desktop/MIR-Project/index/"
        self.document_indexes = {
            Indexes.STARS: Index_reader(path, Indexes.STARS),
            Indexes.GENRES: Index_reader(path, Indexes.GENRES),
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES),
        }
        self.tiered_index = {
            Indexes.STARS: Index_reader(path, Indexes.STARS, Index_types.TIERED),
            Indexes.GENRES: Index_reader(path, Indexes.GENRES, Index_types.TIERED),
            Indexes.SUMMARIES: Index_reader(
                path, Indexes.SUMMARIES, Index_types.TIERED
            ),
        }
        self.document_lengths_index = {
            Indexes.STARS: Index_reader(
                path, Indexes.STARS, Index_types.DOCUMENT_LENGTH
            ),
            Indexes.GENRES: Index_reader(
                path, Indexes.GENRES, Index_types.DOCUMENT_LENGTH
            ),
            Indexes.SUMMARIES: Index_reader(
                path, Indexes.SUMMARIES, Index_types.DOCUMENT_LENGTH
            ),
        }
        self.metadata_index = Index_reader(
            path, Indexes.DOCUMENTS, Index_types.METADATA
        )

    def search(
        self,
        query,
        method,
        weights,
        safe_ranking=True,
        max_results=10,
        smoothing_method=None,
        alpha=0.5,
        lamda=0.5,
    ):
        """
        searches for the query in the indexes.

        Parameters
        ----------
        query : str
            The query to search for.
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25 | Unigram
            The method to use for searching.
        weights: dict
            The weights of the fields.
        safe_ranking : bool
            If True, the search engine will search in whole index and then rank the results.
            If False, the search engine will search in tiered index.
        max_results : int
            The maximum number of results to return. If None, all results are returned.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        list
            A list of tuples containing the document IDs and their scores sorted by their scores.
        """
        preprocessor = Preprocessor([query])
        query = preprocessor.preprocess()[0].split(' ')

        scores = {}
        if method == "unigram":
            self.find_scores_with_unigram_model(
                query, smoothing_method, weights, scores, alpha, lamda
            )
        elif safe_ranking:
            self.find_scores_with_safe_ranking(query, method, weights, scores)
        else:
            self.find_scores_with_unsafe_ranking(
                query, method, weights, max_results, scores
            )

        final_scores = {}

        self.aggregate_scores(weights, scores, final_scores)

        result = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        if max_results is not None:
            result = result[:max_results]

        return result

    def aggregate_scores(self, weights, scores, final_scores):
        """
        Aggregates the scores of the fields.

        Parameters
        ----------
        weights : dict
            The weights of the fields.
        scores : dict
            The scores of the fields.
        final_scores : dict
            The final scores of the documents.
        """
        # TODO
        
        for doc_id, score_dict in scores.items():
            sum = 0
            for field in weights:
                if field in score_dict:
                    sum += score_dict[field] * weights[field]
            final_scores[doc_id] = sum
            

    def find_scores_with_unsafe_ranking(
        self, query, method, weights, max_results, scores
    ):
        """
        Finds the scores of the documents using the unsafe ranking method using the tiered index.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        max_results : int
            The maximum number of results to return.
        scores : dict
            The scores of the documents.
        """
        for tier in ["first_tier", "second_tier", "third_tier"]:
            for field in weights:
                #TODO
                tiered_field_index = self.tiered_index[field].index[tier]
                scorer_obj = Scorer(tiered_field_index, self.metadata_index.index['document_count'])
            
                if method == 'OkapiBM25':
                    results = scorer_obj.compute_socres_with_okapi_bm25(query, self.metadata_index.index['average_document_length'][field.value], self.document_lengths_index[field].index)
                else:
                    results = scorer_obj.compute_scores_with_vector_space_model(query, method)
                    
                self.merge_scores(scores, results, field)
                
            if len(list(scores.keys())) >= max_results:
                break

    def find_scores_with_safe_ranking(self, query, method, weights, scores):
        """
        Finds the scores of the documents using the safe ranking method.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        scores : dict
            The scores of the documents.
        """

        for field in weights:
            #TODO
            field_indexes = self.document_indexes[field].index
            scorer_obj = Scorer(field_indexes, self.metadata_index.index['document_count'])
            
            if method == 'OkapiBM25':
                results = scorer_obj.compute_socres_with_okapi_bm25(query, self.metadata_index.index['average_document_length'][field.value], self.document_lengths_index[field].index)
            else:
                results = scorer_obj.compute_scores_with_vector_space_model(query, method)
                
            self.merge_scores(scores, results, field)

    def find_scores_with_unigram_model(
        self, query, smoothing_method, weights, scores, alpha=0.5, lamda=0.5
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        weights : dict
            A dictionary mapping each field (e.g., 'stars', 'genres', 'summaries') to its weight in the final score. Fields with a weight of 0 are ignored.
        scores : dict
            The scores of the documents.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.
        """
        # TODO
        for field in weights:
            field_indexes = self.document_indexes[field].index
            scorer_obj = Scorer(field_indexes, self.metadata_index.index['document_count'])
            
            results = scorer_obj.compute_scores_with_unigram_model(query, smoothing_method, self.document_lengths_index[field].index, alpha, lamda)
            
            self.merge_scores(scores, results, field)

    def merge_scores(self, scores1, scores2, field):
        """
        Merges two dictionaries of scores.

        Parameters
        ----------
        scores1 : dict
            The first dictionary of scores.
        scores2 : dict
            The second dictionary of scores.

        Returns
        -------
        dict
            The merged dictionary of scores.
        """

        #TODO
        for doc_id, score in scores2.items():
            if doc_id in scores1:
                scores1[doc_id][field] = score
            else:
                scores1[doc_id] = {field: score}

if __name__ == "__main__":
    search_engine = SearchEngine()
    query = "spider man in wonderland"
    method = "unigram"
    weights = {Indexes.STARS: 1, Indexes.GENRES: 1, Indexes.SUMMARIES: 1}
    result = search_engine.search(query, method, weights)

    print(result)
    """"
    with open('/Users/hajmohammadrezaee/Desktop/MIR-Project/Logic/core/utility/validation.json', 'r') as f:
        validation = json.loads(f.read())
        f.close()

    queries = list(validation.keys())
    actual = list(validation.values())

    predicted = {}
    search_engine = SearchEngine()
    method = "lnc.ltc"
    weights = {Indexes.STARS: 1, Indexes.GENRES: 1, Indexes.SUMMARIES: 1}
    result = search_engine.search(query, method, weights)

    print(result)
    """
