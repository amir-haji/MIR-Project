class SpellCorrection:
    def __init__(self, all_documents):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str
            The input documents.
        """
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(all_documents)

    def shingle_word(self, word, k=2):
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """
        shingles = set()
        
        # TODO: Create shingle here
        for i in range(0, len(word) - k + 1):
            shingles.add(word[i: i+k])

        return shingles
    
    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """

        # TODO: Calculate jaccard score here.
        
        jaccard_score = len(first_set.intersection(second_set))/len(first_set.union(second_set))
        return jaccard_score

    def shingling_and_counting(self, all_documents):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_shingled_words = dict()
        word_counter = dict()

        # TODO: Create shingled words dictionary and word counter dictionary here.
        
        for doc_id in all_documents:
            result = ''
            document = all_documents[doc_id]
            for summary in document['summaries']:
                s = summary.strip()
                result = result + s + ' '
                
            result = result.strip() 
            words = result.split(' ')
            
            for word in words:
                if word not in all_shingled_words:
                    all_shingled_words[word] = self.shingle_word(word)
                    word_counter[word] = 1
                else:
                    word_counter[word] += 1
                    
        return all_shingled_words, word_counter
    
    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : stf
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        top5_candidates = list()

        # TODO: Find 5 nearest candidates here.
        
        candidates = []
        shingles = self.shingle_word(word)
        
        for candidate, candidate_shingle in self.all_shingled_words.items():
            score = self.jaccard_score(shingles, candidate_shingle)
            candidates.append([candidate, score])
            
        candidates.sort(key=lambda x: x[1], reverse=True)
        top5_candidates = candidates[:5]
        
        max_tf = -1
        for word, _ in top5_candidates:
            max_tf = max(max_tf, self.word_counter[word])
            
        for x in top5_candidates:
            x[1] = x[1]*(self.word_counter[x[0]])/max_tf
            
        top5_candidates.sort(key=lambda x: x[1], reverse=True)
        top5_candidates = [x[0] for x in top5_candidates]
        return top5_candidates
    
    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : stf
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        final_result = ""
        
        # TODO: Do spell correction here.
        
        query_words = query.split(' ')
        
        for word in query_words:
            if word in self.all_shingled_words:
                final_result = final_result + word + ' '
            else:
                candidates = self.find_nearest_words(word)
                print(candidates)
                if candidates[0] is not None:
                    final_result = final_result + candidates[0] + ' '
                else:
                    final_result = final_result + word + ' '

        return final_result.strip()