import sys
sys.path.append('/Users/hajmohammadrezaee/Desktop/MIR-Project/Logic/core/utility')
from preprocess import Preprocessor

import numpy as np
import itertools
import random
import hashlib
import json

class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.num_hashes = num_hashes
        self.shingled_documents = [self.shingle_document(doc) for doc in documents]
        self.characteristic_matrix = self.build_characteristic_matrix()
        self.signature_matrix = self.min_hash_signature()
        self.bands = 10
        self.rows_per_band = num_hashes // self.bands

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        shingles = set()
        words = document.split()
        for i in range(len(words) - k + 1):
            shingle = ' '.join(words[i:i+k])
            shingles.add(shingle)
        return shingles

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        shingle_list = list(set().union(*self.shingled_documents))
        char_matrix = np.zeros((len(self.shingled_documents), len(shingle_list)), dtype=int)
        for i, doc in enumerate(self.shingled_documents):
            for j, shingle in enumerate(shingle_list):
                if shingle in doc:
                    char_matrix[i, j] = 1
        return char_matrix

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        num_docs, num_shingles = self.characteristic_matrix.shape
        signature_matrix = np.full((self.num_hashes, num_docs), np.inf)

        for i in range(num_shingles):
            hash_values = np.array([hash(f'{j}{i}') % num_shingles for j in range(self.num_hashes)])
            for doc_index in range(num_docs):
                if self.characteristic_matrix[doc_index, i] == 1:
                    for hash_index in range(self.num_hashes):
                        signature_matrix[hash_index, doc_index] = min(signature_matrix[hash_index, doc_index],
                                                                       hash_values[hash_index])
        return signature_matrix

    def lsh_buckets(self):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        bands = self.bands
        rows_per_band = self.rows_per_band
        num_docs = self.signature_matrix.shape[1]
        buckets = {}

        for b in range(bands):
            bucket_dict = {}
            for doc_index in range(num_docs):
                band_hash = hashlib.sha256(self.signature_matrix[b*rows_per_band:(b+1)*rows_per_band, doc_index].tostring()).hexdigest()
                # band_hash = hash(self.signature_matrix[b*rows_per_band:(b+1)*rows_per_band, doc_index].tostring())
                if band_hash not in bucket_dict:
                    bucket_dict[band_hash] = [doc_index]
                else:
                    bucket_dict[band_hash].append(doc_index)

            for key, value in bucket_dict.items():
                if len(value) > 1:
                    if key not in buckets:
                        buckets[key] = []
                    buckets[key].extend(value)

        return buckets

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        self.min_hash_signature()
        return self.lsh_buckets()

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        intersection = len(first_set.intersection(second_set))
        union = len(first_set.union(second_set))
        
        score = intersection / union if union != 0 else 0
        return score

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)

if __name__ == "__main__":
    with open('/Users/hajmohammadrezaee/Desktop/preprocessed.json', 'r') as f:
        data1 = json.loads(f.read())
        f.close()

    with open('/Users/hajmohammadrezaee/Desktop/MIR-Project/Logic/core/indexer/LSHFakeData.json', 'r') as f:
        data2 = json.loads(f.read())
        f.close()

    preprocess_obj = Preprocessor(data2)
    preprocess_obj.preprocess()

    documents = []

    for doc in data1:
        sum = ''
        for summary in doc['summaries']:
            summary = summary.strip()
            sum = sum + summary + ' '
        sum.strip()
        
        documents.append(sum)
    

    for doc in preprocess_obj.documents:
        sum = ''
        for summary in doc['summaries']:
            summary = summary.strip()
            sum = sum + summary + ' '
        sum.strip()
        
        documents.append(sum)

    minhash_obj = MinHashLSH(documents, 50)
    buckets = minhash_obj.perform_lsh()

    empty_summary_key = list(buckets.keys())[0]
    empty_index = buckets[empty_summary_key]
    del buckets[empty_summary_key]

    minhash_obj.jaccard_similarity_test(buckets, documents)

    """
    find the indexes for deleting the near-duplicate and empty documents
    """

    remove_index = empty_index
    for key, value in buckets.items():
        if len(set(value)) > 1:
            combinations = list(itertools.combinations(set(value), 2))
            for comb in combinations:

                first_doc_id = comb[0]
                second_doc_id = comb[1]

                first_shingled_doc = minhash_obj.shingle_document(documents[first_doc_id])
                second_shingled_doc = minhash_obj.shingle_document(documents[second_doc_id])

                jaccard_score = minhash_obj.jaccard_score(first_shingled_doc, second_shingled_doc)
                
                if jaccard_score > 0.6:
                    empty_index.append(second_doc_id)

    remove_index = list(set(remove_index))
    remove_index = sorted(remove_index, reverse=True)

    data = data1 + data2

    for index in remove_index:
        del data[index]

    with open('/Users/hajmohammadrezaee/Desktop/preprocessed_duplicate_checked.json', 'w') as f:
        f.write(json.dumps(data))
        f.close()




    
