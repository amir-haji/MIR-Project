import time
import os
import json
import copy
from indexes_enum import Indexes
from tiered_index import Tiered_index

class Index:
    def __init__(self, preprocessed_documents: list):
        """
        Create a class for indexing.
        """

        self.preprocessed_documents = preprocessed_documents
        self.index_of_documents = self.index_documents()
        
        self.index = {
            Indexes.DOCUMENTS.value: self.index_of_documents,
            Indexes.STARS.value: self.index_stars(),
            Indexes.GENRES.value: self.index_genres(),
            Indexes.SUMMARIES.value: self.index_summaries(),
        }

    def index_documents(self):
        """
        Index the documents based on the document ID. In other words, create a dictionary
        where the key is the document ID and the value is the document.

        Returns
        ----------
        dict
            The index of the documents based on the document ID.
        """

        current_index = {}
        #TODO
        
        for doc in self.preprocessed_documents:
            document_id = doc['id']
            current_index[document_id] = doc

        return current_index

    def index_stars(self):
        """
        Index the documents based on the stars.

        Returns
        ----------
        dict
            The index of the documents based on the stars. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        #         TODO
        current_index = {}
        sorted_ids = sorted(list(self.index_of_documents.keys()))
        for ID in sorted_ids:
            doc = self.index_of_documents[ID]
            for name in doc['stars']:
                terms = name.split(' ')
                for term in terms:
                    if term in current_index:
                        if ID in current_index[term]:
                            current_index[term][ID] += 1
                        else:
                            current_index[term][ID] = 1
                    else:
                        current_index[term] = {ID: 1}
                        
        return current_index

    def index_genres(self):
        """
        Index the documents based on the genres.

        Returns
        ----------
        dict
            The index of the documents based on the genres. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        #         TODO
        current_index = {}
        sorted_ids = sorted(list(self.index_of_documents.keys()))
        for ID in sorted_ids:
            doc = self.index_of_documents[ID]
            for name in doc['genres']:
                terms = name.split(' ')
                for term in terms:
                    if term in current_index:
                        if ID in current_index[term]:
                            current_index[term][ID] += 1
                        else:
                            current_index[term][ID] = 1
                    else:
                        current_index[term] = {ID: 1}
                        
        return current_index

    def index_summaries(self):
        """
        Index the documents based on the summaries (not first_page_summary).

        Returns
        ----------
        dict
            The index of the documents based on the summaries. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        #         TODO
        current_index = {}
        sorted_ids = sorted(list(self.index_of_documents.keys()))
        for ID in sorted_ids:
            doc = self.index_of_documents[ID]
            for summary in doc['summaries']:
                terms = summary.split(' ')
                for term in terms:
                    if term in current_index:
                        if ID in current_index[term]:
                            current_index[term][ID] += 1
                        else:
                            current_index[term][ID] = 1
                    else:
                        current_index[term] = {ID: 1}

        return current_index

    def get_posting_list(self, word: str, index_type: str):
        """
        get posting_list of a word

        Parameters
        ----------
        word: str
            word we want to check
        index_type: str
            type of index we want to check (documents, stars, genres, summaries)

        Return
        ----------
        list
            posting list of the word (you should return the list of document IDs that contain the word and ignore the tf)
        """

        try:
            #         TODO
            return list(self.index[index_type][word].keys())
        except:
            return []

    def add_document_to_index(self, document: dict):
        """
        Add a document to all the indexes

        Parameters
        ----------
        document : dict
            Document to add to all the indexes
        """

        #         TODO
        ID = document['id']
        if ID in self.index[Indexes.DOCUMENTS.value]:
            print('document was existed in index')
        else:
            self.index[Indexes.DOCUMENTS.value][ID] = document
            
            current_index = self.index[Indexes.STARS.value]
            for name in document['stars']:
                terms = name.split(' ')
                for term in terms:
                    if term in current_index:
                        if ID in current_index[term]:
                            current_index[term][ID] += 1
                        else:
                            current_index[term][ID] = 1
                    else:
                        current_index[term] = {ID: 1}
                        
            current_index = self.index[Indexes.GENRES.value]
            for name in document['genres']:
                terms = name.split(' ')
                for term in terms:
                    if term in current_index:
                        if ID in current_index[term]:
                            current_index[term][ID] += 1
                        else:
                            current_index[term][ID] = 1
                    else:
                        current_index[term] = {ID: 1}
                        
            current_index = self.index[Indexes.SUMMARIES.value]
            for summary in document['summaries']:
                terms = summary.split(' ')
                for term in terms:
                    if term in current_index:
                        if ID in current_index[term]:
                            current_index[term][ID] += 1
                        else:
                            current_index[term][ID] = 1
                    else:
                        current_index[term] = {ID: 1}

    def remove_document_from_index(self, document_id: str):
        """
        Remove a document from all the indexes

        Parameters
        ----------
        document_id : str
            ID of the document to remove from all the indexes
        """

        #         TODO
        if document_id not in self.index[Indexes.DOCUMENTS.value]:
            print('document does\'nt exist in index')
        else:
            document = self.index[Indexes.DOCUMENTS.value][document_id]
            
            current_index = self.index[Indexes.STARS.value]
            for name in document['stars']:
                terms = name.split(' ')
                for term in terms:
                    if term in current_index:
                        if document_id in current_index[term]:
                            del current_index[term][document_id]
            
            current_index = self.index[Indexes.GENRES.value]
            for name in document['genres']:
                terms = name.split(' ')
                for term in terms:
                    if term in current_index:
                        if document_id in current_index[term]:
                            del current_index[term][document_id]
                            
            current_index = self.index[Indexes.SUMMARIES.value]
            for summary in document['summaries']:
                terms = summary.split(' ')
                for term in terms:
                    if term in current_index:
                        if document_id in current_index[term]:
                            del current_index[term][document_id]
                            
            del self.index[Indexes.DOCUMENTS.value][document_id]

    def check_add_remove_is_correct(self):
        """
        Check if the add and remove is correct
        """

        dummy_document = {
            'id': '100',
            'stars': ['tim', 'henry'],
            'genres': ['drama', 'crime'],
            'summaries': ['good']
        }

        index_before_add = copy.deepcopy(self.index)
        self.add_document_to_index(dummy_document)
        index_after_add = copy.deepcopy(self.index)

        if index_after_add[Indexes.DOCUMENTS.value]['100'] != dummy_document:
            print('Add is incorrect, document')
            return

        if (set(index_after_add[Indexes.STARS.value]['tim']).difference(set(index_before_add[Indexes.STARS.value]['tim']))
                != {dummy_document['id']}):
            print('Add is incorrect, tim')
            return

        if (set(index_after_add[Indexes.STARS.value]['henry']).difference(set(index_before_add[Indexes.STARS.value]['henry']))
                != {dummy_document['id']}):
            print('Add is incorrect, henry')
            return
        if (set(index_after_add[Indexes.GENRES.value]['drama']).difference(set(index_before_add[Indexes.GENRES.value]['drama']))
                != {dummy_document['id']}):
            print('Add is incorrect, drama')
            return

        if (set(index_after_add[Indexes.GENRES.value]['crime']).difference(set(index_before_add[Indexes.GENRES.value]['crime']))
                != {dummy_document['id']}):
            print('Add is incorrect, crime')
            return

        if (set(index_after_add[Indexes.SUMMARIES.value]['good']).difference(set(index_before_add[Indexes.SUMMARIES.value]['good']))
                != {dummy_document['id']}):
            print('Add is incorrect, good')
            return

        print('Add is correct')

        self.remove_document_from_index('100')
        index_after_remove = copy.deepcopy(self.index)

        if index_after_remove == index_before_add:
            print('Remove is correct')
        else:
            print('Remove is incorrect')

    def store_index(self, path: str, index_name: str = None):
        """
        Stores the index in a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to store the file
        index_name: str
            name of index we want to store (documents, stars, genres, summaries)
        """

        if not os.path.exists(path):
            os.makedirs(path)

        if index_name is None:
            # TODO
            Tiered_index(path)
            return
                        
            
        if index_name not in self.index:
            raise ValueError('Invalid index type')

        #         TODO
        with open(path + f'{index_name}_index.json', 'w') as f:
            f.write(json.dumps(self.index[index_name]))
            f.close()

    def load_index(self, path: str):
        """
        Loads the index from a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to load the file
        """

        #         TODO
        with open(path + '/stars_index.json', 'r') as f:
            self.index[Indexes.STARS.value] = json.loads(f.read())
            f.close()
            
        with open(path + '/genres_index.json', 'r') as f:
            self.index[Indexes.GENRES.value] = json.loads(f.read())
            f.close()
            
        with open(path + '/summaries_index.json', 'r') as f:
            self.index[Indexes.SUMMARIES.value] = json.loads(f.read())
            f.close()

    def check_if_index_loaded_correctly(self, index_type: str, loaded_index: dict):
        """
        Check if the index is loaded correctly

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        loaded_index : dict
            The loaded index

        Returns
        ----------
        bool
            True if index is loaded correctly, False otherwise
        """

        return self.index[index_type] == loaded_index

    def check_if_indexing_is_good(self, index_type: str, check_word: str = 'good'):
        """
        Checks if the indexing is good. Do not change this function. You can use this
        function to check if your indexing is correct.

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        check_word : str
            The word to check in the index

        Returns
        ----------
        bool
            True if indexing is good, False otherwise
        """

        # brute force to check check_word in the summaries
        start = time.time()
        docs = []
        for document in self.preprocessed_documents:
            if index_type not in document or document[index_type] is None:
                continue

            for field in document[index_type]:
                if check_word in field:
                    docs.append(document['id'])
                    break

            # if we have found 3 documents with the word, we can break
            if len(docs) == 3:
                break

        end = time.time()
        brute_force_time = end - start

        # check by getting the posting list of the word
        start = time.time()
        # TODO: based on your implementation, you may need to change the following line
        posting_list = self.get_posting_list(check_word, index_type)

        end = time.time()
        implemented_time = end - start

        print('Brute force time: ', brute_force_time)
        print('Implemented time: ', implemented_time)

        
        if set(docs).issubset(set(posting_list)):
            print('Indexing is correct')

            if implemented_time < brute_force_time:
                print('Indexing is good')
                return True
            else:
                print('Indexing is bad')
                return False
        else:
            print('Indexing is wrong')
            return False

# TODO: Run the class with needed parameters, then run check methods and finally report the results of check methods

if __name__ == "__main__":
    with open('/Users/hajmohammadrezaee/Desktop/preprocessed_duplicate_checked.json', 'r') as f:
        preprocessed_data = json.loads(f.read())
        f.close()

    index_obj = Index(preprocessed_data)

    index_obj.check_add_remove_is_correct()
    index_obj.check_if_indexing_is_good('summaries')
    index_obj.check_if_indexing_is_good('summaries', 'dracula')

    index_obj.check_if_indexing_is_good('genres', 'tv')
    index_obj.check_if_indexing_is_good('genres', 'show')

    index_obj.check_if_indexing_is_good('stars', 'christian')
    index_obj.check_if_indexing_is_good('stars', 'tarantino')

    path = 'index/'
    index_obj.store_index(path, 'documents')
    index_obj.store_index(path, 'genres')
    index_obj.store_index(path, 'stars')
    index_obj.store_index(path, 'summaries')
    
    #######################
    index_obj.load_index('/Users/hajmohammadrezaee/Desktop/MIR-Project/index')

    for index_name in ['stars', 'genres', 'summaries', 'documents']:
        with open(f'/Users/hajmohammadrezaee/Desktop/MIR-Project/index/{index_name}_index.json', 'r') as f:
            index = json.loads(f.read())
            f.close()

        ans = index_obj.check_if_index_loaded_correctly(index_name, index)
        if ans:
            print(f'{index_name} loaded correctly')
        else:
            print(f'{index_name} loaded wrong')
