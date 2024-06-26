from typing import Dict, List
from .core.search import SearchEngine
from .core.utility.spell_correction import SpellCorrection
from .core.utility.snippet import Snippet
from .core.utility.preprocess import Preprocessor
from .core.indexer.indexes_enum import Indexes, Index_types
import json

# TODO: load your movies dataset (from the json file you saved your indexes in), here
# You can refer to `get_movie_by_id` to see how this is used.

with open('/Users/hajmohammadrezaee/Desktop/MIR-Project/index/documents_index.json', 'r') as f:
    all_documents = json.loads(f.read())
    f.close()

search_engine = SearchEngine()

def correct_text(text: str, all_documents: List[str]) -> str:
    """
    Correct the give query text, if it is misspelled using Jacard similarity

    Paramters
    ---------
    text: str
        The query text
    all_documents : list of str
        The input documents.

    Returns
    str
        The corrected form of the given text
    """
    # TODO: You can add any preprocessing steps here, if needed!
    # spell_correction_obj = SpellCorrection(all_documents)
    # text = spell_correction_obj.spell_check(text)

    spell_correction_obj = SpellCorrection(all_documents)
    text = spell_correction_obj.spell_check(text)
    preprocess_obj = Preprocessor([])
    text = preprocess_obj.normalize(text)
    return text


def search(
    query: str,
    max_result_count: int,
    method: str = "ltn-lnn",
    weights: list = [0.3, 0.3, 0.4],
    unigram_smoothing = None,
    should_print=False,
    preferred_genre: str = None,
    smoothing_method = None, 
    alpha=0.5, 
    lamda=0.5
):
    """
    Finds relevant documents to query

    Parameters
    ---------------------------------------------------------------------------------------------------
    query:
        The query text

    max_result_count: Return top 'max_result_count' docs which have the highest scores.
                      notice that if max_result_count = -1, then you have to return all docs

    method: 'ltn.lnn' or 'ltc.lnc' or 'OkapiBM25'

    weights:
        The list, containing importance weights in the search result for each of these items:
            Indexes.STARS: weights[0],
            Indexes.GENRES: weights[1],
            Indexes.SUMMARIES: weights[2],

    preferred_genre:
        A list containing preference rates for each genre. If None, the preference rates are equal.
        (You can leave it None for now)

    Returns
    ----------------------------------------------------------------------------------------------------
    list
    Retrieved documents with snippet
    """
    weights = {
        Indexes.STARS: weights[0],
        Indexes.GENRES: weights[1],
        Indexes.SUMMARIES: weights[2]
    } 
    return search_engine.search(
        query, method, weights, max_results=max_result_count, safe_ranking=True, smoothing_method=unigram_smoothing, alpha=alpha, lamda=lamda
    )


def get_movie_by_id(id: str, movies_dataset: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Get movie by its id

    Parameters
    ---------------------------------------------------------------------------------------------------
    id: str
        The id of the movie

    movies_dataset: List[Dict[str, str]]
        The dataset of movies

    Returns
    ----------------------------------------------------------------------------------------------------
    dict
        The movie with the given id
    """
    print('\%\%*' + id + '\%\%*')
    result = movies_dataset.get(
         id,
         {
             "Title": "This is movie's title",
             "Summary": "This is a summary",
             "URL": "https://www.imdb.com/title/tt0111161/",
             "Cast": ["Morgan Freeman", "Tim Robbins"],
             "Genres": ["Drama", "Crime"],
             "Image_URL": "https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxZDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_.jpg",
         },
     )

    result["Image_URL"] = (
         "https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxZDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_.jpg"  # a default picture for selected movies
     )
    
    print('*' * 10, result, '*' * 10)
    
    result["URL"] = (
         f"https://www.imdb.com/title/{result['id']}"  # The url pattern of IMDb movies
     )
    return result
