class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side
        
        with open('/Users/hajmohammadrezaee/Desktop/MIR-Project/Logic/core/utility/stopwords.txt', 'r') as f:
            self.stopwords = f.read().split('\n')
            f.close()

    def remove_stop_words_from_query(self, query):
        """
        Remove stop words from the input string.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        str
            The query without stop words.
        """

        # TODO: remove stop words from the query.
        q = query
        for stopword in self.stopwords:
            q = q.replace(stopword, '')
        
        return q

    def find_snippet(self, doc, query):
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Sahwshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """
        final_snippet = ""
        not_exist_words = []

        # TODO: Extract snippet and the tokens which are not present in the doc.
        
        query = self.remove_stop_words_from_query(query)
        query_tokens = query.split(' ')
        doc_tokens = doc.split(' ')
        occurance_indexes = []
        
        for token in query_tokens:
            if token not in doc_tokens:
                not_exist_words.append(token)
            else:
                indices = [i for i, x in enumerate(doc_tokens) if x == token]
                occurance_indexes = occurance_indexes + indices
                
        for index in occurance_indexes:
            doc_tokens[index] = '***' + doc_tokens[index] + '***'

        eliminated_indices = []
        for i in range(0, len(occurance_indexes) - 1):
            if occurance_indexes[i+1] - occurance_indexes[i] >= 10:
                eliminated_indices.append(occurance_indexes[i+1])
                        
        for index in occurance_indexes:
            if index not in eliminated_indices:
                sub_snippet = ' '.join(doc_tokens[max(index - self.number_of_words_on_each_side, 0): min(index + self.number_of_words_on_each_side + 1, len(doc_tokens) - 1)])
                print(sub_snippet)
                final_snippet = final_snippet + sub_snippet + '...'

        return final_snippet, not_exist_words
