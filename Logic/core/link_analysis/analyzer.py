from graph import LinkGraph
import sys
sys.path.append('/Users/hajmohammadrezaee/Desktop/MIR-Project/Logic/core')
from indexer.indexes_enum import Indexes
from indexer.index_reader import Index_reader
import json

class LinkAnalyzer:
    def __init__(self, root_set):
        """
        Initialize the Link Analyzer attributes:

        Parameters
        ----------
        root_set: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "title": string of movie title
            "stars": A list of movie star names
        """
        self.root_set = root_set
        self.graph = LinkGraph()
        self.hubs = []
        self.authorities = []
        self.initiate_params()

    def initiate_params(self):
        """
        Initialize links graph, hubs list and authorities list based of root set

        Parameters
        ----------
        This function has no parameters. You can use self to get or change attributes
        """
        for movie in self.root_set:
            id, title, stars = list(movie.values())
            self.graph.add_node(title)
            self.hubs.append(title)
            for star in stars:
                self.graph.add_node(star)
                self.authorities.append(star)
                self.graph.add_edge(title, star)
            

    def expand_graph(self, corpus):
        """
        expand hubs, authorities and graph using given corpus

        Parameters
        ----------
        corpus: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "stars": A list of movie star names

        Note
        ---------
        To build the base set, we need to add the hubs and authorities that are inside the corpus
        and refer to the nodes in the root set to the graph and to the list of hubs and authorities.
        """
        for movie in corpus:
            id, title, stars = list(movie.values())
            
            if title in self.hubs:
                continue

            add_movie = False
            for star in stars:
                if star in self.authorities:
                    add_movie = True
                    break
                    
            if add_movie:
                self.graph.add_node(title)
                self.hubs.append(title)
                for star in stars:
                    if star not in self.authorities:
                        self.graph.add_node(star)
                        self.authorities.append(star)
                    self.graph.add_edge(title, star)
                    
            

    def hits(self, num_iteration=5, max_result=10):
        """
        Return the top movies and actors using the Hits algorithm

        Parameters
        ----------
        num_iteration: int
            Number of algorithm execution iterations
        max_result: int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            List of names of 10 actors with the most scores obtained by Hits algorithm in descending order
        list
            List of names of 10 movies with the most scores obtained by Hits algorithm in descending order
        """
        a_s = dict(zip(self.authorities, [1] * len(self.authorities)))
        h_s = dict(zip(self.hubs, [1] * len(self.hubs)))

        
        for _ in range(num_iteration):
            for star in self.authorities:
                movies = self.graph.get_predecessors(star)
                for movie in movies:
                    h_s[movie] += a_s[star]
                    
            for movie in self.hubs:
                stars = self.graph.get_successors(movie)
                for star in stars:
                    a_s[star] += h_s[movie]
                    
            sum_h = sum(list(h_s.values()))
            sum_a = sum(list(a_s.values()))
            
            a_s.update((x, y/sum_a) for x, y in a_s.items())
            h_s.update((x, y/sum_h) for x, y in h_s.items())
            
        a_s = sorted(a_s.items(), key = lambda x: (x[1], x[0]), reverse = True)[:max_result]
        h_s = sorted(h_s.items(), key = lambda x: (x[1], x[0]), reverse = True)[:max_result]
        
        a_s = [x[0] for x in a_s]
        h_s = [x[0] for x in h_s]
        
        return a_s, h_s



if __name__ == "__main__":
    # You can use this section to run and test the results of your link analyzer
    with open('../IMDB_crawled.json', 'r') as f:
        data = json.loads(f.read())
        f.close() 

    root_set = []
    for movie in data:
        if 'Batman' in movie['title']:
            root_set.append(movie)  
    
    root_set = [{'id': movie['id'],\
                  'title': movie['title'],\
                      'stars': [] if movie['stars'] is None else movie['stars']}\
                          for movie in root_set]
    
    corpus = [{'id': movie['id'],\
                'title': movie['title'],\
                      'stars': [] if movie['stars'] is None else movie['stars']}\
                          for movie in data]

    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=5)
    print("Top Actors:")
    print(*actors, sep=' - ')
    print("Top Movies:")
    print(*movies, sep=' - ')
